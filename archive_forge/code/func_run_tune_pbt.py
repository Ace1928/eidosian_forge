import argparse
import json
import os
import random
import tempfile
import numpy as np
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining
def run_tune_pbt(smoke_test=False):
    perturbation_interval = 5
    pbt = PopulationBasedTraining(time_attr='training_iteration', perturbation_interval=perturbation_interval, hyperparam_mutations={'lr': tune.uniform(0.0001, 0.02), 'some_other_factor': [1, 2]})
    tuner = tune.Tuner(pbt_function, run_config=train.RunConfig(name='pbt_function_api_example', verbose=False, stop={'done': True, 'training_iteration': 10 if smoke_test else 1000}, failure_config=train.FailureConfig(fail_fast=True), checkpoint_config=train.CheckpointConfig(checkpoint_score_attribute='mean_accuracy', num_to_keep=2)), tune_config=tune.TuneConfig(scheduler=pbt, metric='mean_accuracy', mode='max', num_samples=8), param_space={'lr': 0.0001, 'some_other_factor': 1, 'checkpoint_interval': perturbation_interval})
    results = tuner.fit()
    print('Best hyperparameters found were: ', results.get_best_result().config)