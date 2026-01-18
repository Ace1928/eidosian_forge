import os
import warnings
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import torch
import ray
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed
def load_torch_model(saved_model: Union[torch.nn.Module, Dict], model_definition: Optional[torch.nn.Module]=None) -> torch.nn.Module:
    """Loads a PyTorch model from the provided ``saved_model``.

    ``model_definition`` is only used when ``saved_model`` is
    a torch state dict, which will be loaded into ``model_definition``.
    Otherwise, ``model_definition`` is discarded.
    """
    if isinstance(saved_model, torch.nn.Module):
        return saved_model
    elif isinstance(saved_model, dict):
        if not model_definition:
            raise ValueError('Attempting to load torch model from a state_dict, but no `model_definition` was provided.')
        model_definition.load_state_dict(saved_model)
        return model_definition
    else:
        raise ValueError(f'Saved model is of type {type(saved_model)}. The model saved in the checkpoint is expected to be of type `torch.nn.Module`, or a model state dict of type dict.')