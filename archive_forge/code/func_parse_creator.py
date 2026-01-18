import copy
from typing import Any, Callable, Dict, List, Optional, no_type_check
from triad import ParamDict
from triad.collections import Schema
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import get_caller_global_local_vars, to_function, to_instance
from triad.utils.hash import to_uuid
from fugue._utils.interfaceless import parse_output_schema_from_comment
from fugue._utils.registry import fugue_plugin
from fugue.dataframe import DataFrame
from fugue.dataframe.function_wrapper import DataFrameFunctionWrapper
from fugue.exceptions import FugueInterfacelessError
from fugue.extensions.creator.creator import Creator
from .._utils import load_namespace_extensions
@fugue_plugin
def parse_creator(obj: Any) -> Any:
    """Parse an object to another object that can be converted to a Fugue
    :class:`~fugue.extensions.creator.creator.Creator`.

    .. admonition:: Examples

        .. code-block:: python

            from fugue import Creator, FugueWorkflow
            from fugue.plugins import parse_creator
            from triad import to_uuid

            class My(Creator):
                def __init__(self, x):
                    self.x = x

                def create(self) :
                    raise NotImplementedError

                def __uuid__(self) -> str:
                    return to_uuid(super().__uuid__(), self.x)

            @parse_creator.candidate(
                lambda x: isinstance(x, str) and x.startswith("-*"))
            def _parse(obj):
                return My(obj)

            dag = FugueWorkflow()
            dag.create("-*abc").show()
            # == dag.create(My("-*abc")).show()

            dag.run()
    """
    if isinstance(obj, str) and obj in _CREATOR_REGISTRY:
        return _CREATOR_REGISTRY[obj]
    return obj