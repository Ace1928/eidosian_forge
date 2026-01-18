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
def get_retry_history(self, retry_name):
    """Fetch a single retrys history."""
    source, _clone = self._atomdetail_by_name(retry_name, expected_type=models.RetryDetail)
    return self._translate_into_history(source)