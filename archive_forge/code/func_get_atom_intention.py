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
def get_atom_intention(self, atom_name):
    """Gets the intention of an atom given an atoms name."""
    source, _clone = self._atomdetail_by_name(atom_name)
    return source.intention