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
def init_world(self):
    """
        Initializes a new world for the dialog.
        """
    neighborhood_ind = random.randint(0, len(self.neighborhoods) - 1)
    self.neighborhood = self.neighborhoods[neighborhood_ind]
    self.min_x = random.randint(0, self.boundaries[self.neighborhood][0]) * 2
    self.min_y = random.randint(0, self.boundaries[self.neighborhood][1]) * 2
    self.max_x = self.min_x + 3
    self.max_y = self.min_y + 3
    self.location = [random.randint(self.min_x, self.max_x), random.randint(self.min_y, self.max_y), random.randint(0, 3)]
    self.target_location = [random.randint(self.min_x, self.max_x), random.randint(self.min_y, self.max_y), random.randint(0, 3)]
    self.start_location = [self.location[0], self.location[1], self.location[2]]
    map_f = os.path.join(self.dir, '{}_map.json'.format(self.neighborhood))
    with open(map_f) as f:
        data = json.load(f)
        for landmark in data:
            if landmark['x'] * 2 >= self.min_x and landmark['x'] * 2 <= self.max_x and (landmark['y'] * 2 >= self.min_y) and (landmark['y'] * 2 <= self.max_y):
                self.landmarks.append(landmark)
    self.send_location(self.agents[0])
    self.send_map(self.agents[1])