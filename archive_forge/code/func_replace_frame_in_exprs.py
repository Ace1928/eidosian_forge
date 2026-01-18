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
def replace_frame_in_exprs(exprs, old_frame, new_frame):
    """
    Translate input expression replacing an input frame in them.

    Parameters
    ----------
    exprs : dict
        Expressions to translate.
    old_frame : HdkOnNativeDataframe
        An input frame to replace.
    new_frame : HdkOnNativeDataframe
        A new input frame to use.

    Returns
    -------
    dict
        Translated expressions.
    """
    mapper = InputMapper()
    mapper.add_mapper(old_frame, FrameMapper(new_frame))
    res = {}
    for col in exprs.keys():
        res[col] = exprs[col].translate_input(mapper)
    return res