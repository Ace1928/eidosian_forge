import heapq
import inspect
import unittest
from pbr.version import VersionInfo
def finishedWith(self, resource, result=None):
    """Indicate that 'resource' has one less user.

        If there are no more registered users of 'resource' then we trigger
        the `clean` hook, which should do any resource-specific
        cleanup.

        :param resource: A resource returned by
            `TestResourceManager.getResource`.
        :param result: An optional TestResult to report resource changes to.
        """
    self._uses -= 1
    if self._uses == 0:
        self._clean_all(resource, result)
        self._setResource(None)