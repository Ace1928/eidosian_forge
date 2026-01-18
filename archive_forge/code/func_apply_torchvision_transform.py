from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Union
import numpy as np
from ray.air.util.data_batch_conversion import BatchFormat
from ray.air.util.tensor_extensions.utils import _create_possibly_ragged_ndarray
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
def apply_torchvision_transform(array: np.ndarray) -> np.ndarray:
    try:
        tensor = convert_ndarray_to_torch_tensor(array)
        output = self._torchvision_transform(tensor)
    except TypeError:
        output = self._torchvision_transform(array)
    if isinstance(output, torch.Tensor):
        output = output.numpy()
    if not isinstance(output, np.ndarray):
        raise ValueError(f'`TorchVisionPreprocessor` expected your transform to return a `torch.Tensor` or `np.ndarray`, but your transform returned a `{type(output).__name__}` instead.')
    return output