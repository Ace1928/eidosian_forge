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
@fasteners.write_locked
def save_retry_failure(self, retry_name, failed_atom_name, failure):
    """Save subflow failure to retry controller history."""
    source, clone = self._atomdetail_by_name(retry_name, expected_type=models.RetryDetail, clone=True)
    try:
        failures = clone.last_failures
    except exceptions.NotFound:
        exceptions.raise_with_cause(exceptions.StorageFailure, 'Unable to fetch most recent retry failures so new retry failure can be inserted')
    else:
        if failed_atom_name not in failures:
            failures[failed_atom_name] = failure
            self._with_connection(self._save_atom_detail, source, clone)