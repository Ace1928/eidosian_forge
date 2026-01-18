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
def test_createsDictionary(self):
    """
        The C{--auth} command line creates a dictionary mapping supported
        interfaces to the list of credentials checkers that support it.
        """
    options = DummyOptions()
    options.parseOptions(['--auth', 'memory', '--auth', 'anonymous'])
    chd = options['credInterfaces']
    self.assertEqual(len(chd[credentials.IAnonymous]), 1)
    self.assertEqual(len(chd[credentials.IUsernamePassword]), 1)
    chdAnonymous = chd[credentials.IAnonymous][0]
    chdUserPass = chd[credentials.IUsernamePassword][0]
    self.assertTrue(checkers.ICredentialsChecker.providedBy(chdAnonymous))
    self.assertTrue(checkers.ICredentialsChecker.providedBy(chdUserPass))
    self.assertIn(credentials.IAnonymous, chdAnonymous.credentialInterfaces)
    self.assertIn(credentials.IUsernamePassword, chdUserPass.credentialInterfaces)