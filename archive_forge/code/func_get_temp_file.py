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
def get_temp_file(self, obj_id: str, permanent: bool) -> str:
    path = self._path if permanent else self._temp_path
    assert_or_throw(path != '', FugueWorkflowRuntimeError(f'{FUGUE_CONF_WORKFLOW_CHECKPOINT_PATH} is not set'))
    return join(path, obj_id + '.parquet')