from typing import Any, Callable, List, Optional, Type, Union
import pandas as pd
from triad import ParamDict, assert_or_throw
from triad.utils.convert import to_instance
from .._utils.registry import fugue_plugin
from ..exceptions import FuguePluginsRegistrationError
from .execution_engine import (
from .native_execution_engine import NativeExecutionEngine
def register_execution_engine(name_or_type: Union[str, Type], func: Callable, on_dup='overwrite') -> None:
    """Register :class:`~fugue.execution.execution_engine.ExecutionEngine` with
    a given name.

    :param name_or_type: alias of the execution engine, or type of an object that
      can be converted to an execution engine
    :param func: a callable taking |ParamsLikeObject| and ``**kwargs`` and returning an
      :class:`~fugue.execution.execution_engine.ExecutionEngine` instance
    :param on_dup: action on duplicated ``name``. It can be "overwrite", "ignore"
      (not overwriting), defaults to "overwrite".

    .. admonition:: Examples

        Alias registration examples:

        .. code-block:: python

            # create a new engine with name my (overwrites if existed)
            register_execution_engine("my", lambda conf: MyExecutionEngine(conf))

            # 0
            make_execution_engine("my")
            make_execution_engine("my", {"myconfig":"value})

            # 1
            dag = FugueWorkflow()
            dag.create([[0]],"a:int").show()
            dag.run("my", {"myconfig":"value})

            # 2
            fsql('''
            CREATE [[0]] SCHEMA a:int
            PRINT
            ''').run("my")

        Type registration examples:

        .. code-block:: python

            from pyspark.sql import SparkSession
            from fugue_spark import SparkExecutionEngine
            from fugue import fsql

            register_execution_engine(
                SparkSession,
                lambda session, conf: SparkExecutionEngine(session, conf))

            spark_session = SparkSession.builder.getOrCreate()

            fsql('''
            CREATE [[0]] SCHEMA a:int
            PRINT
            ''').run(spark_session)
    """
    if isinstance(name_or_type, str):
        nm = name_or_type
        parse_execution_engine.register(func=lambda engine, conf, **kwargs: func(conf, **kwargs), matcher=lambda engine, conf, **kwargs: isinstance(engine, str) and engine == nm, priority=_get_priority(on_dup))
    else:
        tp = name_or_type
        parse_execution_engine.register(func=lambda engine, conf, **kwargs: func(engine, conf, **kwargs), matcher=lambda engine, conf, **kwargs: isinstance(engine, tp), priority=_get_priority(on_dup))