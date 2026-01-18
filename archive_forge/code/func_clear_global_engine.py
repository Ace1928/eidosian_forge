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
def clear_global_engine() -> None:
    """Remove the global exeuction engine (if set)"""
    _FUGUE_GLOBAL_EXECUTION_ENGINE_CONTEXT.set(None)