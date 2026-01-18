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
def set_atom_state(self, atom_name, state):
    """Sets an atoms state."""
    source, clone = self._atomdetail_by_name(atom_name, clone=True)
    if source.state != state:
        clone.state = state
        self._with_connection(self._save_atom_detail, source, clone)