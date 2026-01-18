from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def concat_episode(self, episode_chunk: 'MultiAgentEpisode') -> None:
    """Adds the given `episode_chunk` to the right side of self.

        For concatenating episodes the following rules hold:
            - IDs are identical.
            - timesteps match (`t` of `self` matches `t_started` of `episode_chunk`).

        Args:
            episode_chunk: `MultiAgentEpsiode` instance that should be concatenated
                to `self`.
        """
    assert episode_chunk.id_ == self.id_
    assert not self.is_done
    assert self.t == episode_chunk.t_started
    observations: MultiAgentDict = {agent_id: agent_obs for agent_id, agent_obs in self.get_observations(global_ts=False).items() if not self.agent_episodes[agent_id].is_done}
    for agent_id, agent_obs in episode_chunk.get_observations(indices=0).items():
        assert agent_id in observations
        assert observations[agent_id] == agent_obs
    for agent_id, agent_eps in self.agent_episodes.items():
        if not agent_eps.is_done:
            agent_eps.concat_episode(episode_chunk.agent_episodes[agent_id])
            assert self.global_t_to_local_t[agent_id][-1] == episode_chunk.global_t_to_local_t[agent_id][0]
            self.global_t_to_local_t[agent_id] += episode_chunk.global_t_to_local_t[agent_id][1:]
            if self.global_actions_t[agent_id][-1] == episode_chunk.global_actions_t[agent_id][0]:
                self.global_actions_t[agent_id] += episode_chunk.global_actions_t[agent_id][1:]
            else:
                self.global_actions_t[agent_id] += episode_chunk.global_actions_t[agent_id]
            indices_for_partial_rewards = episode_chunk.partial_rewards_t[agent_id].find_indices_right(self.t)
            self.partial_rewards_t[agent_id] += list(map(episode_chunk.partial_rewards_t[agent_id].__getitem__, indices_for_partial_rewards))
            self.partial_rewards[agent_id] += list(map(episode_chunk.partial_rewards[agent_id].__getitem__, indices_for_partial_rewards))
    self._copy_buffer(episode_chunk)
    self.t = episode_chunk.t
    if episode_chunk.is_terminated:
        self.is_terminated = True
    if episode_chunk.is_truncated:
        self.is_truncated = True