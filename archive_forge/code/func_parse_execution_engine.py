from typing import Any, Callable, List, Optional, Type, Union
import pandas as pd
from triad import ParamDict, assert_or_throw
from triad.utils.convert import to_instance
from .._utils.registry import fugue_plugin
from ..exceptions import FuguePluginsRegistrationError
from .execution_engine import (
from .native_execution_engine import NativeExecutionEngine
@fugue_plugin
def parse_execution_engine(engine: Any=None, conf: Any=None, **kwargs: Any) -> ExecutionEngine:
    """Parse object as
    :class:`~fugue.execution.execution_engine.ExecutionEngine`

    :param engine: it can be empty string or null (use the default execution
      engine), a string (use the registered execution engine), an
      :class:`~fugue.execution.execution_engine.ExecutionEngine` type, or
      the :class:`~fugue.execution.execution_engine.ExecutionEngine` instance
      , or a tuple of two values where the first value represents execution
      engine and the second value represents the sql engine (you can use ``None``
      for either of them to use the default one), defaults to None
    :param conf: |ParamsLikeObject|, defaults to None
    :param kwargs: additional parameters to initialize the execution engine

    :return: the :class:`~fugue.execution.execution_engine.ExecutionEngine`
      instance

    .. admonition:: Examples

        .. code-block:: python

            register_default_execution_engine(lambda conf: E1(conf))
            register_execution_engine("e2", lambda conf, **kwargs: E2(conf, **kwargs))

            register_sql_engine("s", lambda conf: S2(conf))

            # E1 + E1.default_sql_engine
            make_execution_engine()

            # E2 + E2.default_sql_engine
            make_execution_engine(e2)

            # E1 + S2
            make_execution_engine((None, "s"))

            # E2(conf, a=1, b=2) + S2
            make_execution_engine(("e2", "s"), conf, a=1, b=2)

            # SparkExecutionEngine + SparkSQLEngine
            make_execution_engine(SparkExecutionEngine)
            make_execution_engine(SparkExecutionEngine(spark_session, conf))

            # SparkExecutionEngine + S2
            make_execution_engine((SparkExecutionEngine, "s"))
    """
    if engine is None or (isinstance(engine, str) and engine == ''):
        return NativeExecutionEngine(conf)
    try:
        return to_instance(engine, ExecutionEngine, kwargs=dict(conf=conf, **kwargs))
    except Exception as e:
        raise FuguePluginsRegistrationError(f'Fugue execution engine is not recognized ({engine}, {conf}, {kwargs}). You may need to register a parser for it.') from e