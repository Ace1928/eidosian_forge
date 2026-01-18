from __future__ import annotations
import functools
import gc
import inspect
import json
import os
import pickle
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import h5py
import numpy as np
import tensorflow as tf
from packaging.version import parse
from . import DataCollatorWithPadding, DefaultDataCollator
from .activations_tf import get_tf_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, TFGenerationMixin
from .tf_utils import (
from .utils import (
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
def keras_serializable(cls):
    """
    Decorate a Keras Layer class to support Keras serialization.

    This is done by:

    1. Adding a `transformers_config` dict to the Keras config dictionary in `get_config` (called by Keras at
       serialization time.
    2. Wrapping `__init__` to accept that `transformers_config` dict (passed by Keras at deserialization time) and
       convert it to a config object for the actual layer initializer.
    3. Registering the class as a custom object in Keras (if the Tensorflow version supports this), so that it does not
       need to be supplied in `custom_objects` in the call to `keras.models.load_model`.

    Args:
        cls (a `keras.layers.Layers subclass`):
            Typically a `TF.MainLayer` class in this project, in general must accept a `config` argument to its
            initializer.

    Returns:
        The same class object, with modifications for Keras deserialization.
    """
    initializer = cls.__init__
    config_class = getattr(cls, 'config_class', None)
    if config_class is None:
        raise AttributeError('Must set `config_class` to use @keras_serializable')

    @functools.wraps(initializer)
    def wrapped_init(self, *args, **kwargs):
        config = args[0] if args and isinstance(args[0], PretrainedConfig) else kwargs.pop('config', None)
        if isinstance(config, dict):
            config = config_class.from_dict(config)
            initializer(self, config, *args, **kwargs)
        elif isinstance(config, PretrainedConfig):
            if len(args) > 0:
                initializer(self, *args, **kwargs)
            else:
                initializer(self, config, *args, **kwargs)
        else:
            raise ValueError('Must pass either `config` (PretrainedConfig) or `config` (dict)')
        self._config = config
        self._kwargs = kwargs
    cls.__init__ = wrapped_init
    if not hasattr(cls, 'get_config'):
        raise TypeError('Only use @keras_serializable on keras.layers.Layer subclasses')
    if hasattr(cls.get_config, '_is_default'):

        def get_config(self):
            cfg = super(cls, self).get_config()
            cfg['config'] = self._config.to_dict()
            cfg.update(self._kwargs)
            return cfg
        cls.get_config = get_config
    cls._keras_serializable = True
    if hasattr(keras.utils, 'register_keras_serializable'):
        cls = keras.utils.register_keras_serializable()(cls)
    return cls