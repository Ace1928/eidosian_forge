from minerl.herobraine.hero.spaces import Box, Dict, Discrete, MultiDiscrete, Enum
import collections
import numpy as np
def test_all_flat_map_nested_dict():
    all_spaces = Dict({'a': Dict({'b': Dict({'c': Discrete(10)}), 'd': Discrete(10)}), 'e': Dict({'f': Discrete(10)})})
    x = all_spaces.sample()
    assert_equal_recursive(all_spaces.unmap(all_spaces.flat_map(x)), x)