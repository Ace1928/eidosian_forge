import copy
import dataclasses
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union
import numpy as np
from packaging import version
from ..utils import TensorType, is_torch_available, is_vision_available, logging
from .utils import ParameterFormat, compute_effective_axis_dimension, compute_serialized_parameters_size
def flatten_output_collection_property(self, name: str, field: Iterable[Any]) -> Dict[str, Any]:
    flattened_output = {}
    if name in ['present', 'past_key_values']:
        for idx, t in enumerate(field):
            self._flatten_past_key_values_(flattened_output, name, idx, t)
    else:
        flattened_output = super().flatten_output_collection_property(name, field)
    return flattened_output