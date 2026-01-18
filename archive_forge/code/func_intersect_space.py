import collections
from minerl.herobraine.hero.handler import Handler
import numpy as np
from functools import reduce
from typing import List, Tuple
from minerl.herobraine.hero.spaces import Box, Dict, Enum, MineRLSpace
def intersect_space(space, sample):
    if isinstance(space, Dict):
        new_sample = collections.OrderedDict()
        for key, value in space.spaces.items():
            new_sample[key] = intersect_space(value, sample[key])
        return new_sample
    elif isinstance(space, Enum):
        if sample not in space:
            return space.default
        else:
            return sample
    else:
        return sample