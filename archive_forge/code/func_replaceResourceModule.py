import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def replaceResourceModule(self, value):
    """
        Restore the original resource module to L{sys.modules}.
        """
    if value is None:
        try:
            del sys.modules['resource']
        except KeyError:
            pass
    else:
        sys.modules['resource'] = value