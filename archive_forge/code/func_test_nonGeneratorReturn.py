import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def test_nonGeneratorReturn(self):
    """
        Ensure that C{TypeError} with a message about L{inlineCallbacks} is
        raised when a non-generator returns something other than a generator.
        """

    def _noYield():
        return 5
    _noYield = inlineCallbacks(_noYield)
    self.assertIn('inlineCallbacks', str(self.assertRaises(TypeError, _noYield)))