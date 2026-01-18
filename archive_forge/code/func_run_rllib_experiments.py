import importlib
import json
import os
from pathlib import Path
import re
import sys
import typer
from typing import Optional
import uuid
import yaml
import ray
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.resources import resources_to_json, json_to_resources
from ray.tune.tune import run_experiments
from ray.tune.schedulers import create_scheduler
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.common import CLIArguments as cli
from ray.rllib.common import FrameworkEnum, SupportedFileType
from ray.rllib.common import download_example_file, get_file_type
def run_rllib_experiments(experiments: dict, v: cli.V, vv: cli.VV, framework: str, trace: cli.Trace, ray_num_nodes: cli.RayNumNodes, ray_num_cpus: cli.RayNumCpus, ray_num_gpus: cli.RayNumGpus, ray_object_store_memory: cli.RayObjectStoreMemory, ray_ui: cli.RayUi, ray_address: cli.RayAddress, local_mode: cli.LocalMode, resume: cli.Resume, scheduler: cli.Scheduler, scheduler_config: cli.SchedulerConfig, algo: cli.Algo, callbacks=None):
    """Main training function for the RLlib CLI, whether you've loaded your
    experiments from a config file or from command line options."""
    verbose = 1
    for exp in experiments.values():
        input_ = exp.get('config', {}).get('input')
        if input_ and input_ != 'sampler':
            exp['config']['input'] = _patch_path(input_)
        if not exp.get('env') and (not exp.get('config', {}).get('env')):
            raise ValueError("You either need to provide an --env argument (e.g. 'CartPole-v1') or pass an `env` key with a valid environment to your `config`argument.")
        elif framework is not None:
            exp['config']['framework'] = framework
        if trace:
            if exp['config']['framework'] not in ['tf2']:
                raise ValueError('Must enable framework=tf2 to enable eager tracing.')
            exp['config']['eager_tracing'] = True
        if v:
            exp['config']['log_level'] = 'INFO'
            verbose = 3
        if vv:
            exp['config']['log_level'] = 'DEBUG'
            verbose = 3
    if ray_num_nodes:
        from ray.cluster_utils import Cluster
        cluster = Cluster()
        for _ in range(ray_num_nodes):
            cluster.add_node(num_cpus=ray_num_cpus or 1, num_gpus=ray_num_gpus or 0, object_store_memory=ray_object_store_memory)
        ray.init(address=cluster.address)
    else:
        ray.init(include_dashboard=ray_ui, address=ray_address, object_store_memory=ray_object_store_memory, num_cpus=ray_num_cpus, num_gpus=ray_num_gpus, local_mode=local_mode)
    scheduler_config = json.loads(scheduler_config)
    trials = run_experiments(experiments, scheduler=create_scheduler(scheduler, **scheduler_config), resume=resume, verbose=verbose, concurrent=True, callbacks=callbacks)
    ray.shutdown()
    checkpoints = []
    for trial in trials:
        if trial.checkpoint:
            checkpoints.append(trial.checkpoint)
    if checkpoints:
        from rich import print
        from rich.panel import Panel
        print('\nYour training finished.')
        print('Best available checkpoint for each trial:')
        for cp in checkpoints:
            print(f'  {cp.path}')
        print('\nYou can now evaluate your trained algorithm from any checkpoint, e.g. by running:')
        print(Panel(f'[green]  rllib evaluate {checkpoints[0].path} --algo {algo}'))