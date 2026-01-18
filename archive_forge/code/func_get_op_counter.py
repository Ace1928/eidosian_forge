import json
from typing import Any, List
from urllib import parse
import pathlib
from filelock import FileLock
from ray.workflow.storage.base import Storage
from ray.workflow.storage.filesystem import FilesystemStorageImpl
import ray.cloudpickle
from ray.workflow import serialization_context
def get_op_counter(self):
    with FileLock(str(self._log_dir / '.lock')):
        with open(self._op_counter, 'rb') as f:
            counter = ray.cloudpickle.load(f)
            return counter