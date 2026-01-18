from gymnasium.spaces import Box, MultiDiscrete, Tuple as TupleSpace
import logging
import numpy as np
import random
import time
from typing import Callable, Optional, Tuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import MultiAgentDict, PolicyID, AgentID
@staticmethod
def get_policy_configs_for_game(game_name: str) -> Tuple[dict, Callable[[AgentID], PolicyID]]:
    obs_spaces = {'3DBall': Box(float('-inf'), float('inf'), (8,)), '3DBallHard': Box(float('-inf'), float('inf'), (45,)), 'GridFoodCollector': Box(float('-inf'), float('inf'), (40, 40, 6)), 'Pyramids': TupleSpace([Box(float('-inf'), float('inf'), (56,)), Box(float('-inf'), float('inf'), (56,)), Box(float('-inf'), float('inf'), (56,)), Box(float('-inf'), float('inf'), (4,))]), 'SoccerPlayer': TupleSpace([Box(-1.0, 1.0, (264,)), Box(-1.0, 1.0, (72,))]), 'Goalie': Box(float('-inf'), float('inf'), (738,)), 'Striker': TupleSpace([Box(float('-inf'), float('inf'), (231,)), Box(float('-inf'), float('inf'), (63,))]), 'Sorter': TupleSpace([Box(float('-inf'), float('inf'), (20, 23)), Box(float('-inf'), float('inf'), (10,)), Box(float('-inf'), float('inf'), (8,))]), 'Tennis': Box(float('-inf'), float('inf'), (27,)), 'VisualHallway': Box(float('-inf'), float('inf'), (84, 84, 3)), 'Walker': Box(float('-inf'), float('inf'), (212,)), 'FoodCollector': TupleSpace([Box(float('-inf'), float('inf'), (49,)), Box(float('-inf'), float('inf'), (4,))])}
    action_spaces = {'3DBall': Box(-1.0, 1.0, (2,), dtype=np.float32), '3DBallHard': Box(-1.0, 1.0, (2,), dtype=np.float32), 'GridFoodCollector': MultiDiscrete([3, 3, 3, 2]), 'Pyramids': MultiDiscrete([5]), 'Goalie': MultiDiscrete([3, 3, 3]), 'Striker': MultiDiscrete([3, 3, 3]), 'SoccerPlayer': MultiDiscrete([3, 3, 3]), 'Sorter': MultiDiscrete([3, 3, 3]), 'Tennis': Box(-1.0, 1.0, (3,)), 'VisualHallway': MultiDiscrete([5]), 'Walker': Box(-1.0, 1.0, (39,)), 'FoodCollector': MultiDiscrete([3, 3, 3, 2])}
    if game_name == 'SoccerStrikersVsGoalie':
        policies = {'Goalie': PolicySpec(observation_space=obs_spaces['Goalie'], action_space=action_spaces['Goalie']), 'Striker': PolicySpec(observation_space=obs_spaces['Striker'], action_space=action_spaces['Striker'])}

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return 'Striker' if 'Striker' in agent_id else 'Goalie'
    elif game_name == 'SoccerTwos':
        policies = {'PurplePlayer': PolicySpec(observation_space=obs_spaces['SoccerPlayer'], action_space=action_spaces['SoccerPlayer']), 'BluePlayer': PolicySpec(observation_space=obs_spaces['SoccerPlayer'], action_space=action_spaces['SoccerPlayer'])}

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return 'BluePlayer' if '1_' in agent_id else 'PurplePlayer'
    else:
        policies = {game_name: PolicySpec(observation_space=obs_spaces[game_name], action_space=action_spaces[game_name])}

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return game_name
    return (policies, policy_mapping_fn)