import os
from typing import Optional
import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled
def scale_card_img(card_img):
    return pygame.transform.scale(card_img, (card_img_width, card_img_height))