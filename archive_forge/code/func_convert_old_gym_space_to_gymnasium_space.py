import gymnasium as gym
from typing import Optional
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def convert_old_gym_space_to_gymnasium_space(space) -> gym.Space:
    """Converts an old gym (NOT gymnasium) Space into a gymnasium.Space.

    Args:
        space: The gym.Space to convert to gymnasium.Space.

    Returns:
         The converted gymnasium.space object.
    """
    from ray.rllib.utils.serialization import gym_space_from_dict, gym_space_to_dict
    return gym_space_from_dict(gym_space_to_dict(space))