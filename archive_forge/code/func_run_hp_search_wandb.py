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
def run_hp_search_wandb(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    from ..integrations import is_wandb_available
    if not is_wandb_available():
        raise ImportError('This function needs wandb installed: `pip install wandb`')
    import wandb
    reporting_to_wandb = False
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, WandbCallback):
            reporting_to_wandb = True
            break
    if not reporting_to_wandb:
        trainer.add_callback(WandbCallback())
    trainer.args.report_to = ['wandb']
    best_trial = {'run_id': None, 'objective': None, 'hyperparameters': None}
    sweep_id = kwargs.pop('sweep_id', None)
    project = kwargs.pop('project', None)
    name = kwargs.pop('name', None)
    entity = kwargs.pop('entity', None)
    metric = kwargs.pop('metric', 'eval/loss')
    sweep_config = trainer.hp_space(None)
    sweep_config['metric']['goal'] = direction
    sweep_config['metric']['name'] = metric
    if name:
        sweep_config['name'] = name

    def _objective():
        run = wandb.run if wandb.run else wandb.init()
        trainer.state.trial_name = run.name
        run.config.update({'assignments': {}, 'metric': metric})
        config = wandb.config
        trainer.objective = None
        trainer.train(resume_from_checkpoint=None, trial=vars(config)['_items'])
        if getattr(trainer, 'objective', None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)
            format_metrics = rewrite_logs(metrics)
            if metric not in format_metrics:
                logger.warning(f'Provided metric {metric} not found. This might result in unexpected sweeps charts. The available metrics are {format_metrics.keys()}')
        best_score = False
        if best_trial['run_id'] is not None:
            if direction == 'minimize':
                best_score = trainer.objective < best_trial['objective']
            elif direction == 'maximize':
                best_score = trainer.objective > best_trial['objective']
        if best_score or best_trial['run_id'] is None:
            best_trial['run_id'] = run.id
            best_trial['objective'] = trainer.objective
            best_trial['hyperparameters'] = dict(config)
        return trainer.objective
    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity) if not sweep_id else sweep_id
    logger.info(f'wandb sweep id - {sweep_id}')
    wandb.agent(sweep_id, function=_objective, count=n_trials)
    return BestRun(best_trial['run_id'], best_trial['objective'], best_trial['hyperparameters'])