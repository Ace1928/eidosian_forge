import copy
from collections import Iterable
import numpy as np
from numba import jit, prange
from numba.typed import List
from ray.rllib.examples.env.coin_game_non_vectorized_env import CoinGame
from ray.rllib.utils import override
@jit(nopython=True)
def place_coin(red_pos_i, blue_pos_i, grid_size):
    red_pos_flat = _flatten_index(red_pos_i, grid_size)
    blue_pos_flat = _flatten_index(blue_pos_i, grid_size)
    possible_coin_pos = np.array([x for x in range(9) if x != blue_pos_flat and x != red_pos_flat])
    flat_coin_pos = np.random.choice(possible_coin_pos)
    return _unflatten_index(flat_coin_pos, grid_size)