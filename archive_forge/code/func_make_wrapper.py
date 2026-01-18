import logging
from typing import Any, Callable, List, Optional, Type, TYPE_CHECKING, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Discrete, MultiDiscrete
from ray.rllib.utils.annotations import PublicAPI, DeveloperAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import (
def make_wrapper(fn):
    if session_or_none is not None:
        args_placeholders = []
        kwargs_placeholders = {}
        symbolic_out = [None]

        def call(*args, **kwargs):
            args_flat = []
            for a in args:
                if type(a) is list:
                    args_flat.extend(a)
                else:
                    args_flat.append(a)
            args = args_flat
            if symbolic_out[0] is None:
                with session_or_none.graph.as_default():

                    def _create_placeholders(path, value):
                        if dynamic_shape:
                            if len(value.shape) > 0:
                                shape = (None,) + value.shape[1:]
                            else:
                                shape = ()
                        else:
                            shape = value.shape
                        return tf1.placeholder(dtype=value.dtype, shape=shape, name='.'.join([str(p) for p in path]))
                    placeholders = tree.map_structure_with_path(_create_placeholders, args)
                    for ph in tree.flatten(placeholders):
                        args_placeholders.append(ph)
                    placeholders = tree.map_structure_with_path(_create_placeholders, kwargs)
                    for k, ph in placeholders.items():
                        kwargs_placeholders[k] = ph
                    symbolic_out[0] = fn(*args_placeholders, **kwargs_placeholders)
            feed_dict = dict(zip(args_placeholders, tree.flatten(args)))
            tree.map_structure(lambda ph, v: feed_dict.__setitem__(ph, v), kwargs_placeholders, kwargs)
            ret = session_or_none.run(symbolic_out[0], feed_dict)
            return ret
        return call
    else:
        return fn