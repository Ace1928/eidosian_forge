import functools
import importlib.metadata
import importlib.util
import json
import numbers
import os
import pickle
import shutil
import sys
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union
import numpy as np
from .. import __version__ as version
from ..utils import flatten_dict, is_datasets_available, is_pandas_available, is_torch_available, logging
from ..trainer_callback import ProgressCallback, TrainerCallback  # noqa: E402
from ..trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, IntervalStrategy  # noqa: E402
from ..training_args import ParallelMode  # noqa: E402
from ..utils import ENV_VARS_TRUE_VALUES, is_torch_tpu_available  # noqa: E402
def run_hp_search_sigopt(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import sigopt
    if trainer.args.process_index == 0:
        if importlib.metadata.version('sigopt') >= '8.0.0':
            sigopt.set_project('huggingface')
            experiment = sigopt.create_experiment(name='huggingface-tune', type='offline', parameters=trainer.hp_space(None), metrics=[{'name': 'objective', 'objective': direction, 'strategy': 'optimize'}], parallel_bandwidth=1, budget=n_trials)
            logger.info(f'created experiment: https://app.sigopt.com/experiment/{experiment.id}')
            for run in experiment.loop():
                with run:
                    trainer.objective = None
                    if trainer.args.world_size > 1:
                        if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                            raise RuntimeError('only support DDP Sigopt HPO for ParallelMode.DISTRIBUTED currently.')
                        trainer._hp_search_setup(run.run)
                        torch.distributed.broadcast_object_list(pickle.dumps(trainer.args), src=0)
                        trainer.train(resume_from_checkpoint=None)
                    else:
                        trainer.train(resume_from_checkpoint=None, trial=run.run)
                    if getattr(trainer, 'objective', None) is None:
                        metrics = trainer.evaluate()
                        trainer.objective = trainer.compute_objective(metrics)
                    run.log_metric('objective', trainer.objective)
            best = list(experiment.get_best_runs())[0]
            best_run = BestRun(best.id, best.values['objective'].value, best.assignments)
        else:
            from sigopt import Connection
            conn = Connection()
            proxies = kwargs.pop('proxies', None)
            if proxies is not None:
                conn.set_proxies(proxies)
            experiment = conn.experiments().create(name='huggingface-tune', parameters=trainer.hp_space(None), metrics=[{'name': 'objective', 'objective': direction, 'strategy': 'optimize'}], parallel_bandwidth=1, observation_budget=n_trials, project='huggingface')
            logger.info(f'created experiment: https://app.sigopt.com/experiment/{experiment.id}')
            while experiment.progress.observation_count < experiment.observation_budget:
                suggestion = conn.experiments(experiment.id).suggestions().create()
                trainer.objective = None
                if trainer.args.world_size > 1:
                    if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                        raise RuntimeError('only support DDP Sigopt HPO for ParallelMode.DISTRIBUTED currently.')
                    trainer._hp_search_setup(suggestion)
                    torch.distributed.broadcast_object_list(pickle.dumps(trainer.args), src=0)
                    trainer.train(resume_from_checkpoint=None)
                else:
                    trainer.train(resume_from_checkpoint=None, trial=suggestion)
                if getattr(trainer, 'objective', None) is None:
                    metrics = trainer.evaluate()
                    trainer.objective = trainer.compute_objective(metrics)
                values = [{'name': 'objective', 'value': trainer.objective}]
                obs = conn.experiments(experiment.id).observations().create(suggestion=suggestion.id, values=values)
                logger.info(f'[suggestion_id, observation_id]: [{suggestion.id}, {obs.id}]')
                experiment = conn.experiments(experiment.id).fetch()
            best = list(conn.experiments(experiment.id).best_assignments().fetch().iterate_pages())[0]
            best_run = BestRun(best.id, best.value, best.assignments)
        return best_run
    else:
        for i in range(n_trials):
            trainer.objective = None
            args_main_rank = list(pickle.dumps(trainer.args))
            if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise RuntimeError('only support DDP Sigopt HPO for ParallelMode.DISTRIBUTED currently.')
            torch.distributed.broadcast_object_list(args_main_rank, src=0)
            args = pickle.loads(bytes(args_main_rank))
            for key, value in asdict(args).items():
                if key != 'local_rank':
                    setattr(trainer.args, key, value)
            trainer.train(resume_from_checkpoint=None)
            if getattr(trainer, 'objective', None) is None:
                metrics = trainer.evaluate()
                trainer.objective = trainer.compute_objective(metrics)
        return None