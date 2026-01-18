from minerl.env.malmo import InstanceManager
import minerl
import time
import gym
import numpy as np
import logging
import coloredlogs
from minerl.herobraine.wrappers.vector_wrapper import Vectorized
from minerl.herobraine.env_specs.obtain_specs import ObtainDiamondDebug
from minerl.herobraine.hero.test_spaces import assert_equal_recursive
from minerl.herobraine.wrappers.obfuscation_wrapper import Obfuscated
import minerl.herobraine.envs as envs
import minerl.herobraine
def test_acitons():
    wrapper = envs.MINERL_OBTAIN_TEST_DENSE_OBF_V0
    acts = gen_obtain_debug_actions(wrapper.env_to_wrap.env_to_wrap)
    for act in acts:
        wrapper.wrap_action(act)