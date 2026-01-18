import os
from typing import Optional
import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled
def sum_hand(hand):
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)