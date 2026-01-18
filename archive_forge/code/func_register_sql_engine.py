from typing import Any, Callable, List, Optional, Type, Union
import pandas as pd
from triad import ParamDict, assert_or_throw
from triad.utils.convert import to_instance
from .._utils.registry import fugue_plugin
from ..exceptions import FuguePluginsRegistrationError
from .execution_engine import (
from .native_execution_engine import NativeExecutionEngine
def register_sql_engine(name: str, func: Callable, on_dup='overwrite') -> None:
    """Register :class:`~fugue.execution.execution_engine.SQLEngine` with
    a given name.

    :param name: name of the SQL engine
    :param func: a callable taking
      :class:`~fugue.execution.execution_engine.ExecutionEngine`
      and ``**kwargs`` and returning a
      :class:`~fugue.execution.execution_engine.SQLEngine` instance
    :param on_dup: action on duplicated ``name``. It can be "overwrite", "ignore"
      (not overwriting), defaults to "overwrite".

    .. admonition:: Examples

        .. code-block:: python

            # create a new engine with name my (overwrites if existed)
            register_sql_engine("mysql", lambda engine: MySQLEngine(engine))

            # create execution engine with MySQLEngine as the default
            make_execution_engine(("", "mysql"))

            # create DaskExecutionEngine with MySQLEngine as the default
            make_execution_engine(("dask", "mysql"))

            # default execution engine + MySQLEngine
            with FugueWorkflow() as dag:
                dag.create([[0]],"a:int").show()
            dag.run(("","mysql"))
    """
    nm = name
    parse_sql_engine.register(func=lambda engine, execution_engine, **kwargs: func(execution_engine, **kwargs), matcher=lambda engine, execution_engine, **kwargs: isinstance(engine, str) and engine == nm, priority=_get_priority(on_dup))