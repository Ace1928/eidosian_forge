import atexit
import os
import shutil
import tempfile
import weakref
from fastimport.reftracker import RefTracker
from ... import lru_cache, trace
from . import branch_mapper
from .helpers import single_plural
def lookup_committish(self, committish):
    """Resolve a 'committish' to a revision id.

        :param committish: A "committish" string
        :return: Bazaar revision id
        """
    if not committish.startswith(b':'):
        raise ValueError(committish)
    return self.marks[committish.lstrip(b':')]