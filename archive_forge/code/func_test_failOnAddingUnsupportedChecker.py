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
def test_failOnAddingUnsupportedChecker(self):
    """
        When addChecker is called with a checker that does not implement any
        supported interfaces, it fails.
        """
    options = OptionsForUsernameHashedPassword()
    self.assertRaises(strcred.UnsupportedInterfaces, options.addChecker, self.badChecker)