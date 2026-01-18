from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def get_return(self, consider_buffer=False) -> float:
    """Get the all-agent return.

        Args:
            consider_buffer; Boolean. Defines, if we should also consider
                buffered rewards wehn calculating return. Agents might
                have received partial rewards, i.e. rewards without an
                observation. These are stored to the buffer and added up
                until the next observation is received by an agent.
        Returns: A float. The aggregate return from all agents.
        """
    assert sum((len(agent_map) for agent_map in self.global_t_to_local_t.values())) > 0, "ERROR: Cannot determine return of episode that hasn't started, yet!Call `MultiAgentEpisode.add_env_reset(observations=)` first (after which `get_return(MultiAgentEpisode)` will be 0)."
    env_return = sum((agent_eps.get_return() for agent_eps in self.agent_episodes.values()))
    if consider_buffer:
        buffered_rewards = 0
        for agent_buffer in self.agent_buffers.values():
            if agent_buffer['rewards'].full():
                agent_buffered_rewards = agent_buffer['rewards'].get_nowait()
                buffered_rewards += agent_buffered_rewards
                agent_buffer['rewards'].put_nowait(agent_buffered_rewards)
        return env_return + buffered_rewards
    else:
        return env_return