import copy
import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy
import random
from ray.tune.search.sample import Categorical, Domain, Function, RandomState
from ray.util.annotations import DeveloperAPI, PublicAPI
@DeveloperAPI
def assign_value(spec: Dict, path: Tuple, value: Any):
    """Assigns a value to a nested dictionary.

    Handles the special case of tuples, in which case the tuples
    will be re-constructed to accomodate the updated value.
    """
    parent_spec = None
    parent_key = None
    for k in path[:-1]:
        parent_spec = spec
        parent_key = k
        spec = spec[k]
    key = path[-1]
    if not isinstance(spec, tuple):
        spec[key] = value
    else:
        if parent_spec is None:
            raise ValueError('Cannot assign value to a tuple.')
        assert isinstance(key, int), 'Tuple key must be an int.'
        parent_spec[parent_key] = spec[:key] + (value,) + spec[key + 1:]