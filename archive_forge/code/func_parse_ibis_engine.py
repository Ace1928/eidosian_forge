from abc import abstractmethod
from typing import Any, Callable
import ibis
from fugue import AnyDataFrame, DataFrame, DataFrames, EngineFacet, ExecutionEngine
from fugue._utils.registry import fugue_plugin
from .._compat import IbisTable
@fugue_plugin
def parse_ibis_engine(obj: Any, engine: ExecutionEngine) -> 'IbisEngine':
    if isinstance(obj, IbisEngine):
        return obj
    raise NotImplementedError(f"Ibis execution engine can't be parsed from {obj}. You may need to register a parser for it.")