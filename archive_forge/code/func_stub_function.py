import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
def stub_function(fn):
    """
    A stub function to represent special functions that are meaningless
    outside the context of a CUDA kernel
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        raise NotImplementedError('%s cannot be called from host code' % fn)
    return wrapped