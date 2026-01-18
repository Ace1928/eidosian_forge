import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
import ray
import ray.experimental.tf_utils
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy, PolicyState, PolicySpec
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.error import ERR_MSG_TF_POLICY_CANNOT_SAVE_KERAS_MODEL
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.tf_run_builder import _TFRunBuilder
from ray.rllib.utils.tf_utils import get_gpu_devices
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
@staticmethod
def next_tf_var_scope_name():
    TFPolicy.tf_var_creation_scope_counter += 1
    return f'var_scope_{TFPolicy.tf_var_creation_scope_counter}'