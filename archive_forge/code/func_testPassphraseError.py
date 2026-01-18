import os
import sys
from textwrap import dedent
from twisted.persisted import sob
from twisted.persisted.styles import Ephemeral
from twisted.python import components
from twisted.trial import unittest
def testPassphraseError(self):
    """
        Calling save() with a passphrase is an error.
        """
    p = sob.Persistant(None, 'object')
    self.assertRaises(TypeError, p.save, 'filename.pickle', passphrase='abc')