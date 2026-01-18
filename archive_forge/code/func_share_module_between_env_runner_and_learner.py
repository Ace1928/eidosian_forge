import copy
import dataclasses
import gc
import logging
import tree  # pip install dm_tree
from typing import Any, Dict, List, Optional, Union
import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.dreamerv3.dreamerv3_catalog import DreamerV3Catalog
from ray.rllib.algorithms.dreamerv3.dreamerv3_learner import (
from ray.rllib.algorithms.dreamerv3.utils import do_symlog_obs
from ray.rllib.algorithms.dreamerv3.utils.env_runner import DreamerV3EnvRunner
from ray.rllib.algorithms.dreamerv3.utils.summaries import (
from ray.rllib.core.learner.learner import LearnerHyperparameters
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.models.catalog import MODEL_DEFAULTS
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.numpy import one_hot
from ray.rllib.utils.metrics import (
from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer
from ray.rllib.utils.typing import LearningRateOrSchedule, ResultDict
@property
def share_module_between_env_runner_and_learner(self) -> bool:
    return self.num_learner_workers == 0 and self.num_rollout_workers == 0