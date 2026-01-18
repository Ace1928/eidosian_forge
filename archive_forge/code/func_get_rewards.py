from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def get_rewards(self, indices: Union[int, List[int]]=-1, global_ts: bool=True, as_list: bool=False, partial: bool=True, consider_buffer: bool=True) -> Union[MultiAgentDict, List[MultiAgentDict]]:
    """Gets rewards for all agents that stepped in the last timesteps.

        Note that rewards are only returned for agents that stepped
        during the given index range.

        Args:
            indices: Either a single index or a list of indices. The indices
                can be reversed (e.g. [-1, -2]) or absolute (e.g. [98, 99]).
                This defines the time indices for which the rewards
                should be returned.
            global_ts: Boolean that defines, if the indices should be considered
                environment (`True`) or agent (`False`) steps.

        Returns: A dictionary mapping agent ids to rewards (of different
            timesteps). Only for agents that have stepped (were ready) at a
            timestep, rewards are returned (i.e. not all agent ids are
            necessarily in the keys).
        """
    if global_ts:
        if isinstance(indices, list):
            indices = [self.t - self.ts_carriage_return + idx + 1 if idx < 0 else idx for idx in indices]
        else:
            indices = [self.t - self.ts_carriage_return + indices + 1] if indices < 0 else [indices]
    elif not isinstance(indices, list):
        indices = [indices]
    if not partial and consider_buffer:
        buffered_rewards = {}
        timestep_mapping = {}
        for agent_id, agent_global_t_to_local_t in self.global_t_to_local_t.items():
            if agent_global_t_to_local_t:
                if self.partial_rewards_t[agent_id] and self.partial_rewards_t[agent_id][-1] > agent_global_t_to_local_t[-1] and (self.partial_rewards_t[agent_id][-1] <= max(indices)):
                    indices_at_or_after_last_obs = [agent_global_t_to_local_t[-1]] + sorted([idx for idx in indices if idx > agent_global_t_to_local_t[-1]])
                    buffered_rewards[agent_id], indices_wih_rewards = self.accumulate_partial_rewards(agent_id, indices_at_or_after_last_obs, return_indices=True)
                    timestep_mapping[agent_id] = _IndexMapping(agent_global_t_to_local_t[1:] + list(range(agent_global_t_to_local_t[-1] + 1, agent_global_t_to_local_t[-1] + 1 + len(indices_wih_rewards))))
                else:
                    buffered_rewards[agent_id] = []
                    timestep_mapping[agent_id] = _IndexMapping(agent_global_t_to_local_t[1:])
            elif self.partial_rewards_t[agent_id] and self.partial_rewards_t[agent_id][0] < max(indices):
                buffered_rewards[agent_id], indices_with_rewards = self.accumulate_partial_rewards(agent_id, [0] + sorted(indices), return_indices=True)
                timestep_mapping[agent_id] = _IndexMapping(range(1, len(indices_with_rewards) + 1))
            else:
                buffered_rewards[agent_id] = []
                timestep_mapping[agent_id] = _IndexMapping()
    if partial:
        if as_list:
            return [{agent_id: self.partial_rewards[agent_id][self.partial_rewards_t[agent_id].find_indices([idx], shift=0)[0]] for agent_id, agent_eps in self.agent_episodes.items() if self.partial_rewards_t[agent_id].find_indices([idx], shift=0)} for idx in indices]
        else:
            return {agent_id: list(map(self.partial_rewards[agent_id].__getitem__, self.partial_rewards_t[agent_id].find_indices(indices, shift=0))) for agent_id in self._agent_ids if self.partial_rewards_t[agent_id].find_indices(indices, shift=0)}
    elif consider_buffer:
        return self._getattr_by_index('rewards', indices, global_ts=global_ts, global_ts_mapping=timestep_mapping, buffered_values=buffered_rewards, as_list=as_list)
    else:
        return self._getattr_by_index('rewards', indices, has_initial_value=True, global_ts=global_ts, as_list=as_list, shift=-1)