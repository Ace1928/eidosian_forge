import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
def supports_input(self, input_name: str) -> bool:
    """
        Checks whether the `DummyInputGenerator` supports the generation of the requested input.

        Args:
            input_name (`str`):
                The name of the input to generate.

        Returns:
            `bool`: A boolean specifying whether the input is supported.

        """
    return any((input_name.startswith(supported_input_name) for supported_input_name in self.SUPPORTED_INPUT_NAMES))