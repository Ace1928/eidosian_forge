import functools
import gymnasium as gym
import logging
import importlib.util
import os
from typing import (
import ray
from ray.actor import ActorHandle
from ray.exceptions import RayActorError
from ray.rllib.core.learner import LearnerGroup
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.utils.actor_manager import RemoteCallResults
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.offline import get_dataset_and_shards
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.typing import (
def set_weight(w):
    w.set_weights(weights, global_vars)