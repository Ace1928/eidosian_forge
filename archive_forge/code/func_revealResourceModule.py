import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def revealResourceModule(self, limit):
    """
        Make a L{FakeResourceModule} instance importable at the L{resource}
        name.

        @param limit: The value which will be returned for the hard limit of
            number of open files by the fake resource module's C{getrlimit}
            function.
        """
    sys.modules['resource'] = FakeResourceModule(limit)