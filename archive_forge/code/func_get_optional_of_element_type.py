import inspect
import pathlib
import sys
import typing
from collections import defaultdict
from types import CodeType
from typing import Dict, Iterable, List, Optional
import torch
def get_optional_of_element_type(types):
    """Extract element type, return as `Optional[element type]` from consolidated types.

    Helper function to extracts the type of the element to be annotated to Optional
    from the list of consolidated types and returns `Optional[element type]`.
    TODO: To remove this check once Union support lands.
    """
    elem_type = types[1] if type(None) == types[0] else types[0]
    elem_type = get_type(elem_type)
    return 'Optional[' + elem_type + ']'