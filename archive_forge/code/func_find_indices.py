from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def find_indices(self, indices_to_find: List[int], shift: int=0):
    """Returns global timesteps at which an agent stepped.

        The function returns for a given list of indices the ones
        that are stored in the `IndexMapping`.

        Args:
            indices_to_find: A list of indices that should be
                found in the `IndexMapping`.

        Returns:
            A list of indices at which to find the `indices_to_find`
            in `self`. This could be empty if none of the given
            indices are in `IndexMapping`.
        """
    indices = []
    for num in indices_to_find:
        if num in self and self.index(num) + shift >= 0:
            indices.append(self.index(num) + shift)
    return indices