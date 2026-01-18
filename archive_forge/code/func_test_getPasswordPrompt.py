import sys
from unittest import skipIf
from twisted.conch.error import ConchError
from twisted.conch.test import keydata
from twisted.internet.testing import StringTransport
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
@skipIf(doSkip, skipReason)
def test_getPasswordPrompt(self):
    """
        Get the password using
        L{twisted.conch.client.default.SSHUserAuthClient.getPassword}
        using a different prompt.
        """
    options = ConchOptions()
    client = SSHUserAuthClient(b'user', options, None)
    prompt = b'Give up your password'

    def getpass(p):
        self.assertEqual(p, nativeString(prompt))
        return 'bad password'
    self.patch(default.getpass, 'getpass', getpass)
    d = client.getPassword(prompt)
    d.addCallback(self.assertEqual, b'bad password')
    return d