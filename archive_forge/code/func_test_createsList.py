import os
from io import StringIO
from typing import Sequence, Type
from unittest import skipIf
from zope.interface import Interface
from twisted import plugin
from twisted.cred import checkers, credentials, error, strcred
from twisted.plugins import cred_anonymous, cred_file, cred_unix
from twisted.python import usage
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def test_createsList(self):
    """
        The C{--auth} command line creates a list in the
        Options instance and appends values to it.
        """
    options = DummyOptions()
    options.parseOptions(['--auth', 'memory'])
    self.assertEqual(len(options['credCheckers']), 1)
    options = DummyOptions()
    options.parseOptions(['--auth', 'memory', '--auth', 'memory'])
    self.assertEqual(len(options['credCheckers']), 2)