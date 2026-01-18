from minerl.herobraine.hero.spaces import Box, Dict, Discrete, MultiDiscrete, Enum
import collections
import numpy as np
def test_all_flat_map_numpy_only():
    spaces = [Box(low=-10, high=10, shape=(5,), dtype=np.int64), Discrete(10), Discrete(100), Discrete(1000), Enum('a', 'b', 'c')]
    for space in spaces:
        x = space.sample()
        x_flat = space.flat_map(x)
        assert np.array_equal(x, space.unmap(x_flat))