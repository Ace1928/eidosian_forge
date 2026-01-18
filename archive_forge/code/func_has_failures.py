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
def has_failures(self):
    """Returns true if there are **any** failures in storage."""
    for fail_cache in self._failures.values():
        if fail_cache:
            return True
    return False