import numpy as np
from functools import reduce
from collections import OrderedDict
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.hero import spaces
from minerl.herobraine.wrappers.util import union_spaces, flatten_spaces, intersect_space
from minerl.herobraine.wrapper import EnvWrapper
def create_action_space(self):
    act_list = self.remaining_action_space
    act_list.append(('vector', spaces.Box(low=0.0, high=1.0, shape=[self.action_vector_len], dtype=np.float32)))
    return spaces.Dict(sorted(act_list))