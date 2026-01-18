from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional, Union
from triad import ParamDict, assert_or_throw
from fugue.column import ColumnExpr, SelectColumns, col, lit
from fugue.constants import _FUGUE_GLOBAL_CONF
from fugue.exceptions import FugueInvalidOperation
from ..collections.partition import PartitionSpec
from ..dataframe.dataframe import AnyDataFrame, DataFrame, as_fugue_df
from .execution_engine import (
from .factory import make_execution_engine, try_get_context_execution_engine
from .._utils.registry import fugue_plugin
@fugue_plugin
def as_fugue_engine_df(engine: ExecutionEngine, df: AnyDataFrame, schema: Any=None) -> DataFrame:
    """Convert a dataframe to a Fugue engine dependent DataFrame.
    This function is used internally by Fugue. It is not recommended
    to use

    :param engine: the ExecutionEngine to use, must not be None
    :param df: a dataframe like object
    :param schema: the schema of the dataframe, defaults to None

    :return: the engine dependent DataFrame
    """
    if schema is None:
        fdf = as_fugue_df(df)
    else:
        fdf = as_fugue_df(df, schema=schema)
    return engine.to_df(fdf)