import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def jax_random_normal(loc=0.0, scale=1.0, size=None, **kwargs):
    from jax.random import normal
    if size is None:
        size = ()
    x = normal(jax_random_get_key(), shape=size, **kwargs)
    if scale != 1.0:
        x *= scale
    if loc != 0.0:
        x += loc
    return x