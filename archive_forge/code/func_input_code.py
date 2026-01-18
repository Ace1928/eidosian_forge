import copy
import inspect
import re
from typing import (
from ..exceptions import InvalidOperationError
from ..utils.assertion import assert_or_throw
from ..utils.convert import get_full_type_path
from ..utils.entry_points import load_entry_point
from ..utils.hash import to_uuid
from .dict import IndexedOrderedDict
@property
def input_code(self) -> str:
    """The input parameters code expression"""
    return ''.join((x.code for x in self._params.values()))