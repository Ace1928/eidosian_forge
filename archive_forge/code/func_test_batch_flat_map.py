from minerl.herobraine.hero.spaces import Box, Dict, Discrete, MultiDiscrete, Enum
import collections
import numpy as np
def test_batch_flat_map():
    for space in [Box(shape=[3, 2], low=-2, high=2, dtype=np.float32), Box(shape=[3], low=-2, high=2, dtype=np.float32), Box(shape=[], low=-2, high=2, dtype=np.float32), MultiDiscrete([3, 4]), MultiDiscrete([3]), Discrete(94), Enum('asd', 'sd', 'asdads', 'qweqwe')]:
        _test_batch_map(space)
        _test_batch_map(space, no_op=True)