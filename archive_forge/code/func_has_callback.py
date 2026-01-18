from typing import Any, Dict, Union
from fugue.collections.partition import PartitionCursor, PartitionSpec
from fugue.dataframe import DataFrame, DataFrames
from fugue.execution.execution_engine import ExecutionEngine
from fugue.extensions._utils import validate_input_schema, validate_partition_spec
from fugue.rpc import RPCClient, RPCServer
from triad.collections import ParamDict, Schema
from triad.utils.convert import get_full_type_path
from triad.utils.hash import to_uuid
@property
def has_callback(self) -> bool:
    """Whether this transformer has callback"""
    return '_has_rpc_client' in self.__dict__ and self._has_rpc_client