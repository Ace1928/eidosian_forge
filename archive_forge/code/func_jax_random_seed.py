import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def jax_random_seed(seed=None):
    from jax.random import PRNGKey
    global _JAX_RANDOM_KEY
    if seed is None:
        from random import SystemRandom
        seed = SystemRandom().randint(-2 ** 63, 2 ** 63 - 1)
    _JAX_RANDOM_KEY = PRNGKey(seed)