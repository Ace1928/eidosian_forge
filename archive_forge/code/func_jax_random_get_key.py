import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def jax_random_get_key():
    from jax.random import split
    global _JAX_RANDOM_KEY
    if _JAX_RANDOM_KEY is None:
        jax_random_seed()
    _JAX_RANDOM_KEY, subkey = split(_JAX_RANDOM_KEY)
    return subkey