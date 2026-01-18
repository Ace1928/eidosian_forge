import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
def determineDefaultFunctionName():
    """
    Return the string used by Python as the name for code objects which are
    compiled from interactive input or at the top-level of modules.
    """
    try:
        1 // 0
    except BaseException:
        return traceback.extract_stack()[-2][2]