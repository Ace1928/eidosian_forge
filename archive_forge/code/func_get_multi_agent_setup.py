import copy
import logging
import math
import os
import sys
from typing import (
from packaging import version
import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.learner.learner import LearnerHyperparameters
from ray.rllib.core.learner.learner_group_config import LearnerGroupConfig, ModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import ModuleID, SingleAgentRLModuleSpec
from ray.rllib.core.learner.learner import TorchCompileWhatToCompile
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.atari_wrappers import is_atari
from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils import deep_update, merge_dicts
from ray.rllib.utils.annotations import (
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import NotProvided, from_config
from ray.rllib.utils.gym import (
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.serialization import (
from ray.rllib.utils.torch_utils import TORCH_COMPILE_REQUIRED_VERSION
from ray.rllib.utils.typing import (
from ray.tune.logger import Logger
from ray.tune.registry import get_trainable_cls
from ray.tune.result import TRIAL_INFO
from ray.tune.tune import _Config
def get_multi_agent_setup(self, *, policies: Optional[MultiAgentPolicyConfigDict]=None, env: Optional[EnvType]=None, spaces: Optional[Dict[PolicyID, Tuple[Space, Space]]]=None, default_policy_class: Optional[Type[Policy]]=None) -> Tuple[MultiAgentPolicyConfigDict, Callable[[PolicyID, SampleBatchType], bool]]:
    """Compiles complete multi-agent config (dict) from the information in `self`.

        Infers the observation- and action spaces, the policy classes, and the policy's
        configs. The returned `MultiAgentPolicyConfigDict` is fully unified and strictly
        maps PolicyIDs to complete PolicySpec objects (with all their fields not-None).

        Examples:
        .. testcode::

            import gymnasium as gym
            from ray.rllib.algorithms.ppo import PPOConfig
            config = (
              PPOConfig()
              .environment("CartPole-v1")
              .framework("torch")
              .multi_agent(policies={"pol1", "pol2"}, policies_to_train=["pol1"])
            )
            policy_dict, is_policy_to_train = config.get_multi_agent_setup(
                env=gym.make("CartPole-v1"))
            is_policy_to_train("pol1")
            is_policy_to_train("pol2")

        Args:
            policies: An optional multi-agent `policies` dict, mapping policy IDs
                to PolicySpec objects. If not provided, will use `self.policies`
                instead. Note that the `policy_class`, `observation_space`, and
                `action_space` properties in these PolicySpecs may be None and must
                therefore be inferred here.
            env: An optional env instance, from which to infer the different spaces for
                the different policies. If not provided, will try to infer from
                `spaces`. Otherwise from `self.observation_space` and
                `self.action_space`. If no information on spaces can be infered, will
                raise an error.
            spaces: Optional dict mapping policy IDs to tuples of 1) observation space
                and 2) action space that should be used for the respective policy.
                These spaces were usually provided by an already instantiated remote
                EnvRunner. If not provided, will try to infer from `env`. Otherwise
                from `self.observation_space` and `self.action_space`. If no
                information on spaces can be inferred, will raise an error.
            default_policy_class: The Policy class to use should a PolicySpec have its
                policy_class property set to None.

        Returns:
            A tuple consisting of 1) a MultiAgentPolicyConfigDict and 2) a
            `is_policy_to_train(PolicyID, SampleBatchType) -> bool` callable.

        Raises:
            ValueError: In case, no spaces can be infered for the policy/ies.
            ValueError: In case, two agents in the env map to the same PolicyID
                (according to `self.policy_mapping_fn`), but have different action- or
                observation spaces according to the infered space information.
        """
    policies = copy.deepcopy(policies or self.policies)
    if isinstance(policies, (set, list, tuple)):
        policies = {pid: PolicySpec() for pid in policies}
    env_obs_space = None
    env_act_space = None
    if isinstance(env, ray.actor.ActorHandle):
        env_obs_space, env_act_space = ray.get(env._get_spaces.remote())
    elif env is not None:
        if hasattr(env, 'single_observation_space') and isinstance(env.single_observation_space, gym.Space):
            env_obs_space = env.single_observation_space
        elif hasattr(env, 'observation_space') and isinstance(env.observation_space, gym.Space):
            env_obs_space = env.observation_space
        if hasattr(env, 'single_action_space') and isinstance(env.single_action_space, gym.Space):
            env_act_space = env.single_action_space
        elif hasattr(env, 'action_space') and isinstance(env.action_space, gym.Space):
            env_act_space = env.action_space
    if spaces is not None:
        if env_obs_space is None:
            env_obs_space = spaces.get('__env__', [None])[0]
        if env_act_space is None:
            env_act_space = spaces.get('__env__', [None, None])[1]
    for pid, policy_spec in policies.copy().items():
        if not isinstance(policy_spec, PolicySpec):
            policies[pid] = policy_spec = PolicySpec(*policy_spec)
        if policy_spec.policy_class is None and default_policy_class is not None:
            policies[pid].policy_class = default_policy_class
        if old_gym and isinstance(policy_spec.observation_space, old_gym.Space):
            policies[pid].observation_space = convert_old_gym_space_to_gymnasium_space(policy_spec.observation_space)
        elif policy_spec.observation_space is None:
            if spaces is not None and pid in spaces:
                obs_space = spaces[pid][0]
            elif env_obs_space is not None:
                if isinstance(env, MultiAgentEnv) and hasattr(env, '_obs_space_in_preferred_format') and env._obs_space_in_preferred_format:
                    obs_space = None
                    mapping_fn = self.policy_mapping_fn
                    one_obs_space = next(iter(env_obs_space.values()))
                    if all((s == one_obs_space for s in env_obs_space.values())):
                        obs_space = one_obs_space
                    elif mapping_fn:
                        for aid in env.get_agent_ids():
                            if mapping_fn(aid, None, worker=None) == pid:
                                if obs_space is not None and env_obs_space[aid] != obs_space:
                                    raise ValueError('Two agents in your environment map to the same policyID (as per your `policy_mapping_fn`), however, these agents also have different observation spaces!')
                                obs_space = env_obs_space[aid]
                else:
                    obs_space = env_obs_space
            elif self.observation_space:
                obs_space = self.observation_space
            else:
                raise ValueError(f"`observation_space` not provided in PolicySpec for {pid} and env does not have an observation space OR no spaces received from other workers' env(s) OR no `observation_space` specified in config!")
            policies[pid].observation_space = obs_space
        if old_gym and isinstance(policy_spec.action_space, old_gym.Space):
            policies[pid].action_space = convert_old_gym_space_to_gymnasium_space(policy_spec.action_space)
        elif policy_spec.action_space is None:
            if spaces is not None and pid in spaces:
                act_space = spaces[pid][1]
            elif env_act_space is not None:
                if isinstance(env, MultiAgentEnv) and hasattr(env, '_action_space_in_preferred_format') and env._action_space_in_preferred_format:
                    act_space = None
                    mapping_fn = self.policy_mapping_fn
                    one_act_space = next(iter(env_act_space.values()))
                    if all((s == one_act_space for s in env_act_space.values())):
                        act_space = one_act_space
                    elif mapping_fn:
                        for aid in env.get_agent_ids():
                            if mapping_fn(aid, None, worker=None) == pid:
                                if act_space is not None and env_act_space[aid] != act_space:
                                    raise ValueError('Two agents in your environment map to the same policyID (as per your `policy_mapping_fn`), however, these agents also have different action spaces!')
                                act_space = env_act_space[aid]
                else:
                    act_space = env_act_space
            elif self.action_space:
                act_space = self.action_space
            else:
                raise ValueError(f"`action_space` not provided in PolicySpec for {pid} and env does not have an action space OR no spaces received from other workers' env(s) OR no `action_space` specified in config!")
            policies[pid].action_space = act_space
        if not isinstance(policies[pid].config, AlgorithmConfig):
            assert policies[pid].config is None or isinstance(policies[pid].config, dict)
            policies[pid].config = self.copy(copy_frozen=False).update_from_dict(policies[pid].config or {})
    if self.policies_to_train is not None and (not callable(self.policies_to_train)):
        pols = set(self.policies_to_train)

        def is_policy_to_train(pid, batch=None):
            return pid in pols
    else:
        is_policy_to_train = self.policies_to_train
    return (policies, is_policy_to_train)