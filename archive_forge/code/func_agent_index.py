import random
from collections import defaultdict
import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.collectors.simple_list_collector import (
from ray.rllib.evaluation.collectors.agent_collector import AgentCollector
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.typing import AgentID, EnvID, EnvInfoDict, PolicyID, TensorType
def agent_index(self, agent_id: AgentID) -> int:
    """Get the index of an agent among its environment.

        A new index will be created if an agent is seen for the first time.

        Args:
            agent_id: ID of an agent.

        Returns:
            The index of this agent.
        """
    if agent_id not in self._agent_to_index:
        self._agent_to_index[agent_id] = self._next_agent_index
        self._next_agent_index += 1
    return self._agent_to_index[agent_id]