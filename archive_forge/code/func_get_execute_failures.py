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
def get_execute_failures(self):
    """Get all ``execute`` failures that happened with this flow."""
    return self._get_failures(states.EXECUTE)