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
def load_world(self, world_idx):
    """
        Loads a world into the task when replaying data.
        """
    if world_idx == -1:
        success_worlds = []
        best_world_len = 1000
        best_world_idx = 1000
        for world_idx in range(len(self.data)):
            if not self.replay_bot:
                if 40 < len(self.data[world_idx]['dialog']) < 120:
                    break
            else:
                world = copy.deepcopy(self.data[world_idx])
                is_success, length = self.is_world_success(world)
                if is_success:
                    success_worlds.append((world_idx, length))
                    if len(world['dialog']) < best_world_len:
                        best_world_idx = world_idx
                        best_world_len = length
        print(success_worlds)
        world_idx = best_world_idx
    print(world_idx, len(self.data[world_idx]['dialog']))
    self.world_idx = world_idx
    world = self.data[world_idx]
    self.loaded_world = world
    self.neighborhood = world['neighborhood']
    self.target_location = world['target_location']
    self.start_location = world['start_location']
    self.location = self.start_location
    self.landmarks = world['landmarks']
    self.replay_acts = world['dialog']
    self.boundaries = world['boundaries']
    self.min_x, self.min_y, self.max_x, self.max_y = self.boundaries
    self.send_location(self.agents[0])
    self.send_map(self.agents[1])