import contextlib
import functools
import fasteners
from oslo_utils import reflection
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions
from taskflow import logging
from taskflow.persistence.backends import impl_memory
from taskflow.persistence import models
from taskflow import retry
from taskflow import states
from taskflow import task
from taskflow.utils import misc
@fasteners.read_locked
def get_task_progress_details(self, task_name):
    """Get the progress details of a task given a tasks name.

        :param task_name: task name
        :returns: None if progress_details not defined, else progress_details
                 dict
        """
    source, _clone = self._atomdetail_by_name(task_name, expected_type=models.TaskDetail)
    try:
        return source.meta[META_PROGRESS_DETAILS]
    except KeyError:
        return None