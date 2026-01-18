import heapq
import inspect
import unittest
from pbr.version import VersionInfo
def dirtied(self, resource):
    """Mark the resource as having been 'dirtied'.

        A resource is dirty when it is no longer suitable for use by other
        tests.

        e.g. a shared database that has had rows changed.
        """
    self._dirty = True