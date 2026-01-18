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
def load_tf_weights_from_h5(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    mismatched_layers = []
    with h5py.File(resolved_archive_file, 'r') as sharded_checkpoint_file:
        saved_h5_model_layers_name = set(load_attributes_from_hdf5_group(sharded_checkpoint_file, 'layer_names'))
        missing_layers = list({layer.name for layer in model.layers} - saved_h5_model_layers_name)
        unexpected_layers = list(saved_h5_model_layers_name - {layer.name for layer in model.layers})
        saved_weight_names_set = set()
        symbolic_weights_names = set()
        weight_value_tuples = []
        for layer in model.layers:
            if layer.name in saved_h5_model_layers_name:
                h5_layer_object = sharded_checkpoint_file[layer.name]
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                saved_weights = {}
                for weight_name in load_attributes_from_hdf5_group(h5_layer_object, 'weight_names'):
                    name = '/'.join(weight_name.split('/')[1:])
                    if _prefix is not None:
                        name = _prefix + '/' + name
                    saved_weights[name] = np.asarray(h5_layer_object[weight_name])
                    saved_weight_names_set.add(name)
                for symbolic_weight in symbolic_weights:
                    if _prefix is not None:
                        delimeter = len(_prefix.split('/'))
                        symbolic_weight_name = '/'.join(symbolic_weight.name.split('/')[:delimeter] + symbolic_weight.name.split('/')[delimeter + 1:])
                    else:
                        symbolic_weight_name = '/'.join(symbolic_weight.name.split('/')[1:])
                    saved_weight_value = saved_weights.get(symbolic_weight_name, None)
                    if saved_weight_value is None and symbolic_weight_name.endswith('embeddings:0'):
                        symbolic_weight_name = symbolic_weight_name[:-12] + 'weight:0'
                        saved_weight_value = saved_weights.get(symbolic_weight_name, None)
                    symbolic_weights_names.add(symbolic_weight_name)
                    if saved_weight_value is not None:
                        if K.int_shape(symbolic_weight) != saved_weight_value.shape:
                            try:
                                array = np.reshape(saved_weight_value, K.int_shape(symbolic_weight))
                            except ValueError as e:
                                if ignore_mismatched_sizes:
                                    mismatched_layers.append((symbolic_weight_name, saved_weight_value.shape, K.int_shape(symbolic_weight)))
                                    continue
                                else:
                                    raise e
                        else:
                            array = saved_weight_value
                        weight_value_tuples.append((symbolic_weight, array))
    K.batch_set_value(weight_value_tuples)
    missing_layers.extend(list(symbolic_weights_names - saved_weight_names_set))
    unexpected_layers.extend(list(saved_weight_names_set - symbolic_weights_names))
    return (missing_layers, unexpected_layers, mismatched_layers)