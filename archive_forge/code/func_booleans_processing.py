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
def booleans_processing(config, **kwargs):
    """
    Process the input booleans of each model.

    Args:
        config ([`PretrainedConfig`]):
            The config of the running model.
        **kwargs:
            The boolean parameters

    Returns:
        A dictionary with the proper values for each boolean
    """
    final_booleans = {}
    if 'output_attentions' in kwargs:
        final_booleans['output_attentions'] = kwargs['output_attentions'] if kwargs['output_attentions'] is not None else config.output_attentions
    final_booleans['output_hidden_states'] = kwargs['output_hidden_states'] if kwargs['output_hidden_states'] is not None else config.output_hidden_states
    final_booleans['return_dict'] = kwargs['return_dict'] if kwargs['return_dict'] is not None else config.return_dict
    if 'use_cache' in kwargs:
        final_booleans['use_cache'] = kwargs['use_cache'] if kwargs['use_cache'] is not None else getattr(config, 'use_cache', None)
    return final_booleans