import sys
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, Callable, List, Optional, no_type_check
from adagio.instances import TaskContext
from adagio.specs import InputSpec, OutputSpec, TaskSpec
from triad import ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from fugue._utils.exception import (
from fugue.collections.partition import PartitionSpec
from fugue.collections.yielded import PhysicalYielded
from fugue.dataframe import DataFrame, DataFrames
from fugue.dataframe.array_dataframe import ArrayDataFrame
from fugue.exceptions import FugueWorkflowError
from fugue.execution import ExecutionEngine
from fugue.extensions.creator.convert import _to_creator
from fugue.extensions.outputter.convert import _to_outputter
from fugue.extensions.processor.convert import _to_processor
from fugue.rpc.base import RPCServer
from fugue.workflow._checkpoint import Checkpoint, StrongCheckpoint
from fugue.workflow._workflow_context import FugueWorkflowContext
def reset_traceback(self, limit: int, should_prune: Optional[Callable[[str], bool]]=None) -> None:
    cf = sys._getframe(1)
    self._traceback = frames_to_traceback(cf, limit=limit, should_prune=should_prune)