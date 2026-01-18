import json
import os
import random
import time
import copy
import numpy as np
import pickle
from joblib import Parallel, delayed
from parlai.core.worlds import MultiAgentDialogWorld
from parlai.mturk.core.agents import MTURK_DISCONNECT_MESSAGE
from parlai.mturk.core.worlds import MTurkOnboardWorld
def construct_expanded_array(self, grid, size):
    """
        Constructing a larger attention grid when replaying actions.

        Used when displaying the heat map for the Guide.
        """
    new_grid = np.full((size, size), -1.0)
    new_grid = self.fill_initial(new_grid, grid, size)
    new_grid = self.fill_neighbors(new_grid, size)
    return new_grid