from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def find_indices_between_right_equal(self, threshold_left: int, threshold_right: int, shift: int=0):
    indices = []
    for num in self:
        if num > threshold_right:
            break
        elif num <= threshold_left or self.index(num) + shift < 0:
            continue
        else:
            indices.append(self.index(num) + shift)
    return indices