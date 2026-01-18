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
def test_helpAuthTypeLimitsOutput(self):
    """
        C{--help-auth-type} will display a warning if you get
        help for an authType that does not supply at least one of the
        credential interfaces our application can use.
        """
    options = OptionsForUsernamePassword()
    invalidFactory = None
    for factory in strcred.findCheckerFactories():
        if not options.supportsCheckerFactory(factory):
            invalidFactory = factory
            break
    self.assertNotIdentical(invalidFactory, None)
    newStdout = StringIO()
    options.authOutput = newStdout
    self.assertRaises(SystemExit, options.parseOptions, ['--help-auth-type', 'anonymous'])
    self.assertIn(strcred.notSupportedWarning, newStdout.getvalue())