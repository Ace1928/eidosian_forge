import gym
import json
import numpy as np
from copy import deepcopy
from minerl.herobraine.hero import mc, handlers
from collections import defaultdict, deque
def get_action_pair(self):
    replay_action = self.actions.popleft()
    next_action = self.actions[0] if len(self.actions) > 0 else None
    return (replay_action, next_action)