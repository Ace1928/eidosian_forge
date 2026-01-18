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
def run_hp_search_optuna(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import optuna
    if trainer.args.process_index == 0:

        def _objective(trial, checkpoint_dir=None):
            checkpoint = None
            if checkpoint_dir:
                for subdir in os.listdir(checkpoint_dir):
                    if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                        checkpoint = os.path.join(checkpoint_dir, subdir)
            trainer.objective = None
            if trainer.args.world_size > 1:
                if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                    raise RuntimeError('only support DDP optuna HPO for ParallelMode.DISTRIBUTED currently.')
                trainer._hp_search_setup(trial)
                torch.distributed.broadcast_object_list(pickle.dumps(trainer.args), src=0)
                trainer.train(resume_from_checkpoint=checkpoint)
            else:
                trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
            if getattr(trainer, 'objective', None) is None:
                metrics = trainer.evaluate()
                trainer.objective = trainer.compute_objective(metrics)
            return trainer.objective
        timeout = kwargs.pop('timeout', None)
        n_jobs = kwargs.pop('n_jobs', 1)
        directions = direction if isinstance(direction, list) else None
        direction = None if directions is not None else direction
        study = optuna.create_study(direction=direction, directions=directions, **kwargs)
        study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
        if not study._is_multi_objective():
            best_trial = study.best_trial
            return BestRun(str(best_trial.number), best_trial.value, best_trial.params)
        else:
            best_trials = study.best_trials
            return [BestRun(str(best.number), best.values, best.params) for best in best_trials]
    else:
        for i in range(n_trials):
            trainer.objective = None
            args_main_rank = list(pickle.dumps(trainer.args))
            if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise RuntimeError('only support DDP optuna HPO for ParallelMode.DISTRIBUTED currently.')
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