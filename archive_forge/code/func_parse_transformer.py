import copy
from typing import Any, Callable, Dict, List, Optional, Type, Union, no_type_check
from triad import ParamDict, Schema
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.convert import get_caller_global_local_vars, to_function, to_instance
from triad.utils.hash import to_uuid
from fugue._utils.interfaceless import is_class_method, parse_output_schema_from_comment
from fugue._utils.registry import fugue_plugin
from fugue.dataframe import ArrayDataFrame, DataFrame, DataFrames, LocalDataFrame
from fugue.dataframe.function_wrapper import DataFrameFunctionWrapper
from fugue.exceptions import FugueInterfacelessError
from fugue.extensions.transformer.constants import OUTPUT_TRANSFORMER_DUMMY_SCHEMA
from fugue.extensions.transformer.transformer import CoTransformer, Transformer
from .._utils import (
@fugue_plugin
def parse_transformer(obj: Any) -> Any:
    """Parse an object to another object that can be converted to a Fugue
    :class:`~fugue.extensions.transformer.transformer.Transformer`.

    .. admonition:: Examples

        .. code-block:: python

            from fugue import Transformer, parse_transformer, FugueWorkflow
            from triad import to_uuid

            class My(Transformer):
                def __init__(self, x):
                    self.x = x

                ...

                def __uuid__(self) -> str:
                    return to_uuid(super().__uuid__(), self.x)

            @parse_transformer.candidate(
                lambda x: isinstance(x, str) and x.startswith("-*"))
            def _parse(obj):
                return My(obj)

            dag = FugueWorkflow()
            dag.df([[0]], "a:int").transform("-*abc")
            # ==  dag.df([[0]], "a:int").transform(My("-*abc"))

            dag.run()
    """
    if isinstance(obj, str) and obj in _TRANSFORMER_REGISTRY:
        return _TRANSFORMER_REGISTRY[obj]
    return obj