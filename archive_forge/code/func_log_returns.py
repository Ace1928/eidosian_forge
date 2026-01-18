import uuid
import gymnasium as gym
from typing import Optional
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.env.external_env import ExternalEnv, _ExternalEnvEpisode
from ray.rllib.utils.typing import MultiAgentDict
@PublicAPI
@override(ExternalEnv)
def log_returns(self, episode_id: str, reward_dict: MultiAgentDict, info_dict: MultiAgentDict=None, multiagent_done_dict: MultiAgentDict=None) -> None:
    """Record returns from the environment.

        The reward will be attributed to the previous action taken by the
        episode. Rewards accumulate until the next action. If no reward is
        logged before the next action, a reward of 0.0 is assumed.

        Args:
            episode_id: Episode id returned from start_episode().
            reward_dict: Reward from the environment agents.
            info_dict: Optional info dict.
            multiagent_done_dict: Optional done dict for agents.
        """
    episode = self._get(episode_id)
    for agent, rew in reward_dict.items():
        if agent in episode.cur_reward_dict:
            episode.cur_reward_dict[agent] += rew
        else:
            episode.cur_reward_dict[agent] = rew
    if multiagent_done_dict:
        for agent, done in multiagent_done_dict.items():
            episode.cur_done_dict[agent] = done
    if info_dict:
        episode.cur_info_dict = info_dict or {}