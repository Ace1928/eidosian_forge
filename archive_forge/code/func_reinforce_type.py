import collections
import functools
import numbers
import sys
from torch.utils.data.datapipes._hook_iterator import hook_iterator, _SnapshotState
from typing import (Any, Dict, Iterator, Generic, List, Set, Tuple, TypeVar, Union,
from typing import _eval_type, _tp_cache, _type_check, _type_repr  # type: ignore[attr-defined]
from typing import ForwardRef
from abc import ABCMeta
from typing import _GenericAlias  # type: ignore[attr-defined, no-redef]
def reinforce_type(self, expected_type):
    """
    Reinforce the type for DataPipe instance.

    And the 'expected_type' is required to be a subtype of the original type
    hint to restrict the type requirement of DataPipe instance.
    """
    if isinstance(expected_type, tuple):
        expected_type = Tuple[expected_type]
    _type_check(expected_type, msg="'expected_type' must be a type")
    if not issubtype(expected_type, self.type.param):
        raise TypeError(f"Expected 'expected_type' as subtype of {self.type}, but found {_type_repr(expected_type)}")
    self.type = _DataPipeType(expected_type)
    return self