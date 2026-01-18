import abc
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union
import gymnasium as gym
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
@staticmethod
def from_policy(policy: 'Policy') -> 'ConnectorContext':
    """Build ConnectorContext from a given policy.

        Args:
            policy: Policy

        Returns:
            A ConnectorContext instance.
        """
    return ConnectorContext(config=policy.config, initial_states=policy.get_initial_state(), observation_space=policy.observation_space, action_space=policy.action_space, view_requirements=policy.view_requirements, is_policy_recurrent=policy.is_recurrent())