from abc import ABC, abstractmethod
from ctypes import ArgumentError
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
from transformers.utils import is_tf_available
from ..base import ExportConfig
@property
def inputs_specs(self) -> List['TensorSpec']:
    """
        List containing the input specs for each input name in self.inputs.

        Returns:
            `List[tf.TensorSpec]`: A list of tensor specs.
        """
    dummy_inputs = self.generate_dummy_inputs()
    return [tf.TensorSpec(dummy_input.shape, dtype=dummy_input.dtype, name=input_name) for input_name, dummy_input in dummy_inputs.items()]