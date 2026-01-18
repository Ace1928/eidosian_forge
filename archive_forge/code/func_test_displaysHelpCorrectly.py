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
def test_displaysHelpCorrectly(self):
    """
        The C{--help-auth-for} argument will correctly display the help file
        for a particular authentication plugin.
        """
    newStdout = StringIO()
    options = DummyOptions()
    options.authOutput = newStdout
    self.assertRaises(SystemExit, options.parseOptions, ['--help-auth-type', 'file'])
    for line in cred_file.theFileCheckerFactory.authHelp:
        if line.strip():
            self.assertIn(line.strip(), newStdout.getvalue())