import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def jax_random_uniform(low=0.0, high=1.0, size=None, **kwargs):
    from jax.random import uniform
    if size is None:
        size = ()
    return uniform(jax_random_get_key(), shape=size, minval=low, maxval=high, **kwargs)