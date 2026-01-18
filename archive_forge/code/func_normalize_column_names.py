from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
from triad.collections.schema import Schema
from triad.utils.rename import normalize_names
from .._utils.registry import fugue_plugin
from .dataframe import AnyDataFrame, DataFrame, as_fugue_df
def normalize_column_names(df: AnyDataFrame) -> Tuple[AnyDataFrame, Dict[str, Any]]:
    """A generic function to normalize any dataframe's column names to follow
    Fugue naming rules

    .. note::

        This is a temporary solution before
        :class:`~triad:triad.collections.schema.Schema`
        can take arbitrary names

    .. admonition:: Examples

        * ``[0,1]`` => ``{"_0":0, "_1":1}``
        * ``["1a","2b"]`` => ``{"_1a":"1a", "_2b":"2b"}``
        * ``["*a","-a"]`` => ``{"_a":"*a", "_a_1":"-a"}``

    :param df: a dataframe object
    :return: the renamed dataframe and the rename operations as a dict that
        can **undo** the change

    .. seealso::

        * :func:`~.get_column_names`
        * :func:`~.rename`
        * :func:`~triad:triad.utils.rename.normalize_names`
    """
    cols = get_column_names(df)
    names = normalize_names(cols)
    if len(names) == 0:
        return (df, {})
    undo = {v: k for k, v in names.items()}
    return (rename(df, names), undo)