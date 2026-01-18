import abc
import collections
from collections import abc as cabc
import itertools
from oslo_utils import reflection
from taskflow.types import sets
from taskflow.utils import misc
def pre_revert(self):
    """Code to be run prior to reverting the atom.

        This works the same as :meth:`.pre_execute`, but for the revert phase.
        """