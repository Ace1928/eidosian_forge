from typing import Any, Callable, List, Optional, Type, Union
import pandas as pd
from triad import ParamDict, assert_or_throw
from triad.utils.convert import to_instance
from .._utils.registry import fugue_plugin
from ..exceptions import FuguePluginsRegistrationError
from .execution_engine import (
from .native_execution_engine import NativeExecutionEngine
@fugue_plugin
def parse_sql_engine(engine: Any=None, execution_engine: Optional[ExecutionEngine]=None, **kwargs: Any) -> SQLEngine:
    """Create :class:`~fugue.execution.execution_engine.SQLEngine`
    with specified ``engine``

    :param engine: it can be empty string or null (use the default SQL
      engine), a string (use the registered SQL engine), an
      :class:`~fugue.execution.execution_engine.SQLEngine` type, or
      the :class:`~fugue.execution.execution_engine.SQLEngine` instance
      (you can use ``None`` to use the default one), defaults to None
    :param execution_engine: the
      :class:`~fugue.execution.execution_engine.ExecutionEngine` instance
      to create
      the :class:`~fugue.execution.execution_engine.SQLEngine`. Normally you
      should always provide this value.
    :param kwargs: additional parameters to initialize the sql engine

    :return: the :class:`~fugue.execution.execution_engine.SQLEngine`
      instance

    .. note::

        For users, you normally don't need to call this function directly.
        Use ``make_execution_engine`` instead

    .. admonition:: Examples

        .. code-block:: python

            register_default_sql_engine(lambda conf: S1(conf))
            register_sql_engine("s2", lambda conf: S2(conf))

            engine = NativeExecutionEngine()

            # S1(engine)
            make_sql_engine(None, engine)

            # S1(engine, a=1)
            make_sql_engine(None, engine, a=1)

            # S2(engine)
            make_sql_engine("s2", engine)
    """
    if engine is None or (isinstance(engine, str) and engine == ''):
        assert_or_throw(execution_engine is not None, ValueError('execution_engine must be provided'))
        return execution_engine.sql_engine
    try:
        return to_instance(engine, SQLEngine, kwargs=dict(execution_engine=execution_engine, **kwargs))
    except Exception as e:
        raise FuguePluginsRegistrationError(f'Fugue SQL engine is not recognized ({engine}, {kwargs}). You may need to register a parser for it.') from e