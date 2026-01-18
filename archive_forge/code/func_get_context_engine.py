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
def get_context_engine() -> ExecutionEngine:
    """Get the execution engine in the current context. Regarding the order of the logic
    please read :func:`~.fugue.execution.factory.make_execution_engine`
    """
    engine = try_get_context_execution_engine()
    if engine is None:
        raise FugueInvalidOperation('No global/context engine is set')
    return engine