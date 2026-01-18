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
def load_tf_shard(model, model_layer_map, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    """
    Loads a shard from a sharded checkpoint file. Handles the missing keys and unexpected keys.

    Args:
        model (`keras.models.Model`): Model in which the weights are loaded
        model_layer_map (`Dict`): A dictionary mapping the layer name to the index of the layer in the model.
        resolved_archive_file (`str`): Path to the checkpoint file from which the weights will be loaded
        ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`): Whether to ignore the mismatched keys

    Returns:
        `keras.models.Model`: Three lists, one for the layers that were found and succesfully restored (from the
        shard file), one for the mismatched layers, and another one for the unexpected layers.
    """
    saved_weight_names_set = set()
    saved_weights = {}
    mismatched_keys = set()
    unexpected_keys = set()
    try:
        with h5py.File(resolved_archive_file, 'r') as sharded_checkpoint_file:
            saved_h5_model_layers_name = set(load_attributes_from_hdf5_group(sharded_checkpoint_file, 'layer_names'))
            weight_value_tuples = []
            for layer_name in saved_h5_model_layers_name:
                h5_layer_object = sharded_checkpoint_file[layer_name]
                saved_weights[layer_name] = np.asarray(h5_layer_object)
                saved_weight_names_set.add(layer_name)
                if layer_name not in model_layer_map:
                    unexpected_keys.add(layer_name)
                else:
                    symbolic_weight = model.weights[model_layer_map[layer_name]]
                    saved_weight_value = saved_weights[layer_name]
                    if saved_weight_value is not None:
                        if K.int_shape(symbolic_weight) != saved_weight_value.shape:
                            try:
                                array = np.reshape(saved_weight_value, K.int_shape(symbolic_weight))
                            except ValueError as e:
                                if ignore_mismatched_sizes:
                                    mismatched_keys.add((layer_name, saved_weight_value.shape, K.int_shape(symbolic_weight)))
                                    continue
                                else:
                                    raise e
                        else:
                            array = saved_weight_value
                    weight_value_tuples.append((symbolic_weight, array))
        K.batch_set_value(weight_value_tuples)
        return (saved_weight_names_set, unexpected_keys, mismatched_keys)
    except Exception as e:
        try:
            with open(resolved_archive_file) as f:
                if f.read().startswith('version'):
                    raise OSError('You seem to have cloned a repository without having git-lfs installed. Please install git-lfs and run `git lfs install` followed by `git lfs pull` in the folder you cloned.')
                else:
                    raise ValueError(f'Unable to locate the file {resolved_archive_file} which is necessary to load this pretrained model. Make sure you have saved the model properly.') from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(f"Unable to load weights from TF checkpoint file for '{resolved_archive_file}' at '{resolved_archive_file}'. If you tried to load a TF model from a sharded checkpoint, you should try converting the model by loading it in pytorch and saving it localy. A convertion script should be realeased soon.")