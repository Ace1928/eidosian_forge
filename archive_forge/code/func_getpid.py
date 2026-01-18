import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def getpid(self):
    """
        Fake os.getpid, always return the same thing
        """
    return 123