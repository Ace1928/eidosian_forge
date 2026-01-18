import copy
from typing import Any, Callable, Dict, List, Optional, no_type_check
from triad import ParamDict, to_uuid
from triad.utils.convert import get_caller_global_local_vars, to_function, to_instance
from fugue._utils.registry import fugue_plugin
from fugue.dataframe import DataFrames
from fugue.dataframe.function_wrapper import DataFrameFunctionWrapper
from fugue.exceptions import FugueInterfacelessError
from fugue.extensions._utils import (
from fugue.extensions.outputter.outputter import Outputter
@fugue_plugin
def parse_outputter(obj: Any) -> Any:
    """Parse an object to another object that can be converted to a Fugue
    :class:`~fugue.extensions.outputter.outputter.Outputter`.

    .. admonition:: Examples

        .. code-block:: python

            from fugue import Outputter, parse_outputter, FugueWorkflow
            from triad import to_uuid

            class My(Outputter):
                def __init__(self, x):
                    self.x = x

                def process(self, dfs):
                    raise NotImplementedError

                def __uuid__(self) -> str:
                    return to_uuid(super().__uuid__(), self.x)

            @parse_outputter.candidate(
                lambda x: isinstance(x, str) and x.startswith("-*"))
            def _parse(obj):
                return My(obj)

            dag = FugueWorkflow()
            dag.df([[0]], "a:int").output("-*abc")
            # ==  dag.df([[0]], "a:int").output(My("-*abc"))

            dag.run()
    """
    if isinstance(obj, str) and obj in _OUTPUTTER_REGISTRY:
        return _OUTPUTTER_REGISTRY[obj]
    return obj