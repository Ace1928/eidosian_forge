from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def get_observations(self, indices: Union[int, List[int]]=-1, global_ts: bool=True, as_list: bool=False) -> Union[MultiAgentDict, List[MultiAgentDict]]:
    """Gets observations for all agents that stepped in the last timesteps.

        Note that observations are only returned for agents that stepped
        during the given index range.

        Args:
            indices: Either a single index or a list of indices. The indices
                can be reversed (e.g. [-1, -2]) or absolute (e.g. [98, 99]).
                This defines the time indices for which the observations
                should be returned.
            global_ts: Boolean that defines, if the indices should be considered
                environment (`True`) or agent (`False`) steps.

        Returns: A dictionary mapping agent ids to observations (of different
            timesteps). Only for agents that have stepped (were ready) at a
            timestep, observations are returned (i.e. not all agent ids are
            necessarily in the keys).
        """
    return self._getattr_by_index('observations', indices, has_initial_value=True, global_ts=global_ts, as_list=as_list)