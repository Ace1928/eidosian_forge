from typing import Any, Callable, List, Optional, Type, Union
import pandas as pd
from triad import ParamDict, assert_or_throw
from triad.utils.convert import to_instance
from .._utils.registry import fugue_plugin
from ..exceptions import FuguePluginsRegistrationError
from .execution_engine import (
from .native_execution_engine import NativeExecutionEngine
def register_default_execution_engine(func: Callable, on_dup='overwrite') -> None:
    """Register :class:`~fugue.execution.execution_engine.ExecutionEngine` as the
    default engine.

    :param func: a callable taking |ParamsLikeObject| and ``**kwargs`` and returning an
      :class:`~fugue.execution.execution_engine.ExecutionEngine` instance
    :param on_dup: action on duplicated ``name``. It can be "overwrite", "ignore"
      (not overwriting), defaults to "overwrite".

    .. admonition:: Examples

        .. code-block:: python

            # create a new engine with name my (overwrites if existed)
            register_default_execution_engine(lambda conf: MyExecutionEngine(conf))

            # the following examples will use MyExecutionEngine

            # 0
            make_execution_engine()
            make_execution_engine(None, {"myconfig":"value})

            # 1
            dag = FugueWorkflow()
            dag.create([[0]],"a:int").show()
            dag.run(None, {"myconfig":"value})

            # 2
            fsql('''
            CREATE [[0]] SCHEMA a:int
            PRINT
            ''').run("", {"myconfig":"value})
    """
    parse_execution_engine.register(func=lambda engine, conf, **kwargs: func(conf, **kwargs), matcher=lambda engine, conf, **kwargs: engine is None or (isinstance(engine, str) and engine == ''), priority=_get_priority(on_dup))