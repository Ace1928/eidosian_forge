from contextlib import closing
from io import StringIO
from os import path
from typing import Optional
import numpy as np
from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled
def get_surf_loc(self, map_loc):
    return ((map_loc[1] * 2 + 1) * self.cell_size[0], (map_loc[0] + 1) * self.cell_size[1])