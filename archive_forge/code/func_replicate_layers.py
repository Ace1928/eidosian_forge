from __future__ import annotations
import copy
import logging
import re
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Optional, Union
import torch
from accelerate.hooks import AlignDevicesHook
from accelerate.utils import named_module_tensors, offload_state_dict
from torch import nn
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from peft.utils import INCLUDE_LINEAR_LAYERS_SHORTHAND
from ..config import PeftConfig
from ..utils import ModulesToSaveWrapper, _get_submodules
def replicate_layers(model: nn.Module, layer_map: list[tuple[int, int]]):
    """Replicate layers in a transfomer model with weight sharing.

    This function looks for a module list attribute at model[(.model)*].layers and replicates the layers in the module
    list according to the layer map. For example the map `[[0, 4], [2, 5]]` will take the set of layers `[0, 1, 2, 3,
    4]` and replace them with a module list containing `[0, 1, 2, 3, 2, 3, 4]`.
    """
    while hasattr(model, 'model'):
        model = model.model
    if hasattr(model, 'bert'):
        model = model.bert
    model_type = None
    layers: nn.ModuleList = None
    if hasattr(model, 'layers'):
        model_type = 'llama'
        layers = model.layers
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        model_type = 'bert'
        layers = model.encoder.layer
    elif hasattr(model, 'h'):
        model_type = 'falcon'
        layers = model.h
    if not model_type or not isinstance(layers, nn.ModuleList):
        raise ValueError('Could not locate the layers attribute in the model. Expected Llama, Bert or Falcon compatible architectures.')
    new_layers = []
    for start, end in layer_map:
        for i in range(start, end):
            current_idx = len(new_layers)
            new_layers.append(clone_module(layers[i], share_weights=True))
            for submodule in new_layers[-1].modules():
                if hasattr(submodule, 'layer_idx'):
                    submodule.layer_idx = current_idx
    layers = nn.ModuleList(new_layers)
    if model_type == 'llama':
        model.layers = layers
    elif model_type == 'bert':
        model.encoder.layer = layers
    elif model_type == 'falcon':
        model.h = layers
    else:
        raise ValueError('Unexpected model type, need to handle post-processing of layers.')
    if hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = len(new_layers)