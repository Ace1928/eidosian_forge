from typing import Any, Callable, List, Optional, Type, Union
import pandas as pd
from triad import ParamDict, assert_or_throw
from triad.utils.convert import to_instance
from .._utils.registry import fugue_plugin
from ..exceptions import FuguePluginsRegistrationError
from .execution_engine import (
from .native_execution_engine import NativeExecutionEngine
def make_execution_engine(engine: Any=None, conf: Any=None, infer_by: Optional[List[Any]]=None, **kwargs: Any) -> ExecutionEngine:
    """Create :class:`~fugue.execution.execution_engine.ExecutionEngine`
    with specified ``engine``

    :param engine: it can be empty string or null (use the default execution
      engine), a string (use the registered execution engine), an
      :class:`~fugue.execution.execution_engine.ExecutionEngine` type, or
      the :class:`~fugue.execution.execution_engine.ExecutionEngine` instance
      , or a tuple of two values where the first value represents execution
      engine and the second value represents the sql engine (you can use ``None``
      for either of them to use the default one), defaults to None
    :param conf: |ParamsLikeObject|, defaults to None
    :param infer_by: List of objects that can be used to infer the execution
      engine using :func:`~.infer_execution_engine`
    :param kwargs: additional parameters to initialize the execution engine

    :return: the :class:`~fugue.execution.execution_engine.ExecutionEngine`
      instance

    .. note::

        This function finds/constructs the engine in the following order:

        * If ``engine`` is None, it first try to see if there is any defined
          context engine to use (=> engine)
        * If ``engine`` is still empty, then it will try to get the global execution
          engine. See
          :meth:`~fugue.execution.execution_engine.ExecutionEngine.set_global`
        * If ``engine`` is still empty, then if ``infer_by``
          is given, it will try to infer the execution engine (=> engine)
        * If ``engine`` is still empty, then it will construct the default
          engine defined by :func:`~.register_default_execution_engine`  (=> engine)
        * Now, ``engine`` must not be empty, if it is an object other than
          :class:`~fugue.execution.execution_engine.ExecutionEngine`, we will use
          :func:`~.parse_execution_engine` to construct (=> engine)
        * Now, ``engine`` must have been an ExecutionEngine object. We update its
          SQL engine if specified, then update its config using ``conf`` and ``kwargs``

    .. admonition:: Examples

        .. code-block:: python

            register_default_execution_engine(lambda conf: E1(conf))
            register_execution_engine("e2", lambda conf, **kwargs: E2(conf, **kwargs))

            register_sql_engine("s", lambda conf: S2(conf))

            # E1 + E1.create_default_sql_engine()
            make_execution_engine()

            # E2 + E2.create_default_sql_engine()
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

            # assume object e2_df can infer E2 engine
            make_execution_engine(infer_by=[e2_df])  # an E2 engine

            # global
            e_global = E1(conf)
            e_global.set_global()
            make_execution_engine()  # e_global

            # context
            with E2(conf).as_context() as ec:
                make_execution_engine()  # ec
            make_execution_engine()  # e_global
    """
    if engine is None:
        engine = try_get_context_execution_engine()
        if engine is None and infer_by is not None:
            engine = infer_execution_engine(infer_by)
    if isinstance(engine, tuple):
        execution_engine = make_execution_engine(engine[0], conf=conf, **kwargs)
        sql_engine = make_sql_engine(engine[1], execution_engine)
        execution_engine.set_sql_engine(sql_engine)
        return execution_engine
    if isinstance(engine, ExecutionEngine):
        result = engine
    else:
        result = parse_execution_engine(engine, conf, **kwargs)
        sql_engine = make_sql_engine(None, result)
        result.set_sql_engine(sql_engine)
    result.conf.update(conf, on_dup=ParamDict.OVERWRITE)
    result.conf.update(kwargs, on_dup=ParamDict.OVERWRITE)
    return result