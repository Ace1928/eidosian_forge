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
def ensure_atom(self, atom):
    """Ensure there is an atomdetail for the **given** atom.

        Returns the uuid for the atomdetail that corresponds to the given atom.
        """
    return self.ensure_atoms([atom])[0]