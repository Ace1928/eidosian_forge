import abc
from typing import TYPE_CHECKING, Dict, List, Union
import numpy as np
import pandas
import pyarrow as pa
from pandas.core.dtypes.common import is_string_dtype
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import EMPTY_ARROW_TABLE, ColNameCodec, get_common_arrow_type
from .db_worker import DbTable
from .expr import InputRefExpr, LiteralExpr, OpExpr
def add_mapper(self, frame, mapper):
    """
        Register a mapper for a frame.

        Parameters
        ----------
        frame : HdkOnNativeDataframe
            A frame for which a mapper is registered.
        mapper : object
            A mapper to register.
        """
    self._mappers[frame] = mapper