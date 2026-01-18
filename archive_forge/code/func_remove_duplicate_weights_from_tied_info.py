import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
import onnx
from onnx import ModelProto
from ..utils import logging
from .transformations_utils import (
def remove_duplicate_weights_from_tied_info(onnx_model: ModelProto, torch_model: 'nn.Module', tied_params: List[List[str]], save_path: str):
    """
    Tries to remove potential duplicate ONNX initializers from the tied information in tied_params.

    Args:
        onnx_model (`onnx.ModelProto`):
            The ONNX model for which to tie potentially duplicate initializers.
        torch_model (`nn.Module`):
            The PyTorch model corresponding to the ONNX one.
        tied_params (`List[List[str]]`):
            A list of groups of torch parameters that are tied, i.e. shared. For them,
            the torch module shares the same pointer.
    """
    tied_params_with_op, tied_groups_to_tie, tied_groups_ignored = _get_weights_to_tie(tied_params, torch_model)
    if len(tied_groups_ignored) >= 1:
        logger.info(f'The groups of weights {tied_groups_ignored} will not be tied as either already tied or tying is not implemented.')
    initializer_name_to_idx = {}
    for idx, initializer in enumerate(onnx_model.graph.initializer):
        initializer_name_to_idx[initializer.name] = idx
    tied_groups_map = _find_matching_initializers(tied_params_with_op, onnx_model, initializer_name_to_idx)
    onnx_model = _deduplicate_gather_matmul(onnx_model, tied_groups_to_tie, tied_groups_map, initializer_name_to_idx)
    check_and_save_model(onnx_model, save_path=save_path)
    return onnx_model