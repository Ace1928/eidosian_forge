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
def get_revert_result(self, atom_name):
    """Gets the ``revert`` results for an atom from storage."""
    try:
        results = self._get(atom_name, 'revert_results', 'revert_failure', _REVERT_STATES_WITH_RESULTS, states.REVERT)
    except exceptions.DisallowedAccess as e:
        if e.state == states.IGNORE:
            exceptions.raise_with_cause(exceptions.NotFound, "Result for atom '%s' revert is not known (as it was ignored)" % atom_name)
        else:
            exceptions.raise_with_cause(exceptions.NotFound, "Result for atom '%s' revert is not known" % atom_name)
    else:
        return results