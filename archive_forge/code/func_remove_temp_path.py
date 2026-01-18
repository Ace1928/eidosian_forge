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
def remove_temp_path(self):
    if self._temp_path != '':
        try:
            rm(self._temp_path, recursive=True)
        except Exception as e:
            self._log.info('Unable to remove ' + self._temp_path, e)