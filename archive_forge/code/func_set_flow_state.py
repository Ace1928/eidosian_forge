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
def set_flow_state(self, state):
    """Set flow details state and save it."""
    source, clone = self._fetch_flowdetail(clone=True)
    clone.state = state
    self._with_connection(self._save_flow_detail, source, clone)