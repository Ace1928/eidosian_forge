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
def cleanup_retry_history(self, retry_name, state):
    """Cleanup history of retry atom with given name."""
    source, clone = self._atomdetail_by_name(retry_name, expected_type=models.RetryDetail, clone=True)
    clone.state = state
    clone.results = []
    self._with_connection(self._save_atom_detail, source, clone)