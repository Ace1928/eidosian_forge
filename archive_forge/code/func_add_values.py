import collections
import logging
import numpy as np
from typing import List, Any, Dict, Optional, TYPE_CHECKING
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.typing import PolicyID, AgentID
from ray.util.debug import log_once
@DeveloperAPI
def add_values(self, agent_id: AgentID, policy_id: AgentID, **values: Any) -> None:
    """Add the given dictionary (row) of values to this batch.

        Args:
            agent_id: Unique id for the agent we are adding values for.
            policy_id: Unique id for policy controlling the agent.
            values: Row of values to add for this agent.
        """
    if agent_id not in self.agent_builders:
        self.agent_builders[agent_id] = SampleBatchBuilder()
        self.agent_to_policy[agent_id] = policy_id
    if agent_id != _DUMMY_AGENT_ID:
        values['agent_id'] = agent_id
    self.agent_builders[agent_id].add_values(**values)