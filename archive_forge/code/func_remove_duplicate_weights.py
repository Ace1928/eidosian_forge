import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
import onnx
from onnx import ModelProto
from ..utils import logging
from .transformations_utils import (
def remove_duplicate_weights(model: ModelProto, inplace: bool=False) -> ModelProto:
    """
    Finds and removes duplicate weights in a model by keeping only unique weights, and make the duplicate values point
    to them.

    This function only removes duplicate weights that are exactly identical (e.g., not transposed).

    Args:
        model (`onnx.ModelProto`): The model to remove duplicates from.
        inplace (`bool`, defaults to False): Whether to perform this transformation inplace.

    Returns:
        `onnx.ModelProto`: The model without duplicates.
    """
    if not inplace:
        model = copy.deepcopy(model)
    duplicates = _find_duplicate_initializers(models=[model])
    name_sharing_dict = _create_name_sharing_dict(duplicates)
    _replace_input_names(models=[model], name_sharing_dict=name_sharing_dict)
    _remove_redundant_initializers(models=[model], name_sharing_dict=name_sharing_dict)
    return model