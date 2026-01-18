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
@contextmanager
def engine_context(engine: AnyExecutionEngine=None, engine_conf: Any=None, infer_by: Optional[List[Any]]=None) -> Iterator[ExecutionEngine]:
    """Make an execution engine and set it as the context engine. This function
    is thread safe and async safe.

    :param engine: an engine like object, defaults to None
    :param engine_conf: the configs for the engine, defaults to None
    :param infer_by: a list of objects to infer the engine, defaults to None

    .. note::

        For more details, please read
        :func:`~.fugue.execution.factory.make_execution_engine`

    .. admonition:: Examples

        .. code-block:: python

            import fugue.api as fa

            with fa.engine_context(spark_session):
                transform(df, func)  # will use spark in this transformation

    """
    e = make_execution_engine(engine, engine_conf, infer_by=infer_by)
    return e._as_context()