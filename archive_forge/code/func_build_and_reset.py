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
def build_and_reset(self, episode: Optional[Episode]=None) -> MultiAgentBatch:
    """Returns the accumulated sample batches for each policy.

        Any unprocessed rows will be first postprocessed with a policy
        postprocessor. The internal state of this builder will be reset.

        Args:
            episode (Optional[Episode]): The Episode object that
                holds this MultiAgentBatchBuilder object or None.

        Returns:
            MultiAgentBatch: Returns the accumulated sample batches for each
                policy.
        """
    self.postprocess_batch_so_far(episode)
    policy_batches = {}
    for policy_id, builder in self.policy_builders.items():
        if builder.count > 0:
            policy_batches[policy_id] = builder.build_and_reset()
    old_count = self.count
    self.count = 0
    return MultiAgentBatch.wrap_as_needed(policy_batches, old_count)