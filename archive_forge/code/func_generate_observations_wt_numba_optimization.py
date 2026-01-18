import copy
from collections import Iterable
import numpy as np
from numba import jit, prange
from numba.typed import List
from ray.rllib.examples.env.coin_game_non_vectorized_env import CoinGame
from ray.rllib.utils import override
@jit(nopython=True)
def generate_observations_wt_numba_optimization(batch_size, red_pos, blue_pos, coin_pos, red_coin, grid_size):
    obs = np.zeros((batch_size, grid_size, grid_size, 4))
    for i in prange(batch_size):
        obs[i, red_pos[i][0], red_pos[i][1], 0] = 1
        obs[i, blue_pos[i][0], blue_pos[i][1], 1] = 1
        if red_coin[i]:
            obs[i, coin_pos[i][0], coin_pos[i][1], 2] = 1
        else:
            obs[i, coin_pos[i][0], coin_pos[i][1], 3] = 1
    return obs