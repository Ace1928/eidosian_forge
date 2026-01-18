from pathlib import Path
from typing import Union
import numpy as np
from collections import OrderedDict
from minerl.herobraine.hero import spaces
from minerl.herobraine.wrappers.vector_wrapper import Vectorized
from minerl.herobraine.wrapper import EnvWrapper
import copy
import os
def make_func(np_lays):

    def func(x):
        for t, data in np_lays:
            if t == 'linear':
                W, b = data
                x = x.dot(W.T) + b
            elif t == 'relu':
                x = x * (x > 0)
            elif t == 'subset_softmax':
                discrete_subset = data
                for a, b in discrete_subset:
                    y = x[..., a:b]
                    e_x = np.exp(y - np.max(x))
                    x[..., a:b] = e_x / e_x.sum(axis=-1)
            else:
                raise NotImplementedError()
        return x
    return func