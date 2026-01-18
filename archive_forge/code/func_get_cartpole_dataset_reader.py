from collections import Counter
import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Tuple as GymTuple
import inspect
import logging
import numpy as np
import os
import pprint
import random
import re
import time
import tree  # pip install dm_tree
from typing import (
import yaml
import ray
from ray import air, tune
from ray.rllib.env.wrappers.atari_wrappers import is_atari, wrap_deepmind
from ray.rllib.utils.framework import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.tune import CLIReporter, run_experiments
def get_cartpole_dataset_reader(batch_size: int=1) -> 'DatasetReader':
    """Returns a DatasetReader for the cartpole dataset.
    Args:
        batch_size: The batch size to use for the reader.
    Returns:
        A rllib DatasetReader for the cartpole dataset.
    """
    from ray.rllib.algorithms import AlgorithmConfig
    from ray.rllib.offline import IOContext
    from ray.rllib.offline.dataset_reader import DatasetReader, get_dataset_and_shards
    path = 'tests/data/cartpole/large.json'
    input_config = {'format': 'json', 'paths': path}
    dataset, _ = get_dataset_and_shards(AlgorithmConfig().offline_data(input_='dataset', input_config=input_config))
    ioctx = IOContext(config=AlgorithmConfig().training(train_batch_size=batch_size).offline_data(actions_in_input_normalized=True), worker_index=0)
    reader = DatasetReader(dataset, ioctx)
    return reader