from typing import Any
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.utils.io import exists, join, makedirs, rm
from fugue.collections.partition import PartitionSpec
from fugue.collections.yielded import PhysicalYielded
from fugue.constants import FUGUE_CONF_WORKFLOW_CHECKPOINT_PATH
from fugue.dataframe import DataFrame
from fugue.exceptions import FugueWorkflowCompileError, FugueWorkflowRuntimeError
from fugue.execution.execution_engine import ExecutionEngine
def init_temp_path(self, execution_id: str) -> str:
    if self._path == '':
        self._temp_path = ''
        return ''
    self._temp_path = join(self._path, execution_id)
    makedirs(self._temp_path, exist_ok=True)
    return self._temp_path