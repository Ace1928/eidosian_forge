import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def getrlimit(self, no):
    """
        A fake of L{resource.getrlimit} which returns a pre-determined result.
        """
    if no == self.RLIMIT_NOFILE:
        return [0, self._limit]
    return [123, 456]