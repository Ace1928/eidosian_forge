from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
from triad.collections.dict import IndexedOrderedDict
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.pandas_like import PD_UTILS
from triad.utils.pyarrow import (
from triad.utils.schema import (
def is_like(self, other: Any, equal_groups: Optional[List[List[Callable[[pa.DataType], bool]]]]=None) -> bool:
    """Check if the two schemas are equal or similar

        :param other: a schema like object
        :param equal_groups: a list of list of functions to check if two types
            are equal, default None

        :return: True if the two schemas are equal

        .. admonition:: Examples

            .. code-block:: python

                s = Schema("a:int,b:str")
                assert s.is_like("a:int,b:str")
                assert not s.is_like("a:long,b:str")
                assert s.is_like("a:long,b:str", equal_groups=[(pa.types.is_integer,)])
        """
    if other is None:
        return False
    if other is self:
        return True
    if isinstance(other, Schema):
        _other = other
    elif isinstance(other, str):
        if equal_groups is None:
            return self.__repr__() == other
        _other = Schema(other)
    else:
        _other = Schema(other)
    return pa_schemas_equal(self.pa_schema, _other.pa_schema, equal_groups=equal_groups)