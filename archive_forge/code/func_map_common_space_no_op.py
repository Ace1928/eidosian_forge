import collections
from functools import reduce
from minerl.herobraine.hero.spaces import Dict
from minerl.herobraine.hero.test_spaces import assert_equal_recursive
import minerl.herobraine.wrappers as wrappers
import minerl.herobraine.envs as envs
from minerl.herobraine.wrappers.util import union_spaces
import numpy as np
def map_common_space_no_op(common_envs):
    np.random.seed(42)
    for base_env in common_envs:
        s = base_env.observation_space.no_op()
        us = base_env.unwrap_observation(s)