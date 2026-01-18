from __future__ import annotations
import functools
from typing import Callable, Dict, List, Sequence, Tuple, Union
import torch
from functorch._C import dim as _C
from ._parsing import (
def composition_to_dims(composition: Sequence[Union[List[Union[str, AnonymousAxis]], str]]) -> List[Union[str, Tuple[str, ...]]]:
    """Convert a `ParsedExpression.composition` into a `Tensor.__getitem__` index of strings representing first
        class dims."""
    dim_composition: List[Union[str, Tuple[str, ...]]] = []
    for dimension in composition:
        if isinstance(dimension, list):
            dim_composition.append(tuple((dim for identifier in dimension for dim in identifier_dim_map[identifier])))
        elif dimension == _ellipsis:
            dim_composition.extend(identifier_dim_map[_ellipsis])
        else:
            raise ValueError(f'Unexpected dimension: {dimension}')
    return dim_composition