import abc
import collections
from collections import abc as cabc
import itertools
from oslo_utils import reflection
from taskflow.types import sets
from taskflow.utils import misc
def post_revert(self):
    """Code to be run after reverting the atom.

        This works the same as :meth:`.post_execute`, but for the revert phase.
        """