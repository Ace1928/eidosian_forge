import copy
from collections import Iterable
import numpy as np
from numba import jit, prange
from numba.typed import List
from ray.rllib.examples.env.coin_game_non_vectorized_env import CoinGame
from ray.rllib.utils import override
@jit(nopython=True)
def compute_reward(batch_size, red_pos, blue_pos, coin_pos, red_coin, asymmetric, both_players_can_pick_the_same_coin):
    reward_red = np.zeros(batch_size)
    reward_blue = np.zeros(batch_size)
    generate = np.zeros(batch_size, dtype=np.bool_)
    red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue = (0, 0, 0, 0)
    for i in prange(batch_size):
        red_first_if_both = None
        if not both_players_can_pick_the_same_coin:
            if _same_pos(red_pos[i], coin_pos[i]) and _same_pos(blue_pos[i], coin_pos[i]):
                red_first_if_both = bool(np.random.randint(0, 1))
        if red_coin[i]:
            if _same_pos(red_pos[i], coin_pos[i]) and (red_first_if_both is None or red_first_if_both):
                generate[i] = True
                reward_red[i] += 1
                if asymmetric:
                    reward_red[i] += 3
                red_pick_any += 1
                red_pick_red += 1
            if _same_pos(blue_pos[i], coin_pos[i]) and (red_first_if_both is None or not red_first_if_both):
                generate[i] = True
                reward_red[i] += -2
                reward_blue[i] += 1
                blue_pick_any += 1
        else:
            if _same_pos(red_pos[i], coin_pos[i]) and (red_first_if_both is None or red_first_if_both):
                generate[i] = True
                reward_red[i] += 1
                reward_blue[i] += -2
                if asymmetric:
                    reward_red[i] += 3
                red_pick_any += 1
            if _same_pos(blue_pos[i], coin_pos[i]) and (red_first_if_both is None or not red_first_if_both):
                generate[i] = True
                reward_blue[i] += 1
                blue_pick_any += 1
                blue_pick_blue += 1
    reward = [reward_red, reward_blue]
    return (reward, generate, red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue)