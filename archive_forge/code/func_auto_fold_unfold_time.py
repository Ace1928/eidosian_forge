import functools
from typing import Optional
import numpy as np
from ray.rllib.core.models.base import Model
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.specs.checker import SpecCheckingError
from ray.rllib.core.models.specs.specs_base import Spec
from ray.rllib.core.models.specs.specs_base import TensorSpec
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.torch.primitives import TorchCNNTranspose, TorchMLP
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
def auto_fold_unfold_time(input_spec: str):
    """Automatically folds/unfolds the time dimension of a tensor.

    This is useful when calling the model requires a batch dimension only, but the
    input data has a batch- and a time-dimension. This decorator will automatically
    fold the time dimension into the batch dimension before calling the model and
    unfold the batch dimension back into the time dimension after calling the model.

    Args:
        input_spec: The input spec of the model.

    Returns:
        A decorator that automatically folds/unfolds the time_dimension if present.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(self, input_data, **kwargs):
            if not hasattr(self, input_spec):
                raise ValueError('The model must have an input_specs attribute to automatically fold/unfold the time dimension.')
            if not torch.is_tensor(input_data):
                raise ValueError(f'input_data must be a torch.Tensor to fold/unfold time automatically, but got {type(input_data)}.')
            actual_shape = list(input_data.shape)
            spec = getattr(self, input_spec)
            try:
                spec.validate(input_data)
            except ValueError as original_error:
                b, t = actual_shape[:2]
                other_dims = actual_shape[2:]
                reshaped_b = b * t
                new_shape = tuple([reshaped_b] + other_dims)
                reshaped_inputs = input_data.reshape(new_shape)
                try:
                    spec.validate(reshaped_inputs)
                except ValueError as new_error:
                    raise SpecCheckingError(f'Attempted to call {func} with input data of shape {actual_shape}. RLlib attempts to automatically fold/unfold the time dimension because {actual_shape} does not match the input spec {spec}. In an attempt to fold the time dimensions to possibly fit the input specs of {func}, RLlib has calculated the new shape {new_shape} and reshaped the input data to {reshaped_inputs}. However, the input data still does not match the input spec. \nOriginal error: \n{original_error}. \nNew error: \n{new_error}.')
                outputs = func(self, reshaped_inputs, **kwargs)
                return outputs.reshape((b, t) + tuple(outputs.shape[1:]))
            return func(self, input_data, **kwargs)
        return wrapper
    return decorator