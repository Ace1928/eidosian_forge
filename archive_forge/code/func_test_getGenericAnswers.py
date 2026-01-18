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
def test_getGenericAnswers(self):
    """
        L{twisted.conch.client.default.SSHUserAuthClient.getGenericAnswers}
        """
    options = ConchOptions()
    client = SSHUserAuthClient(b'user', options, None)

    def getpass(prompt):
        self.assertEqual(prompt, 'pass prompt')
        return 'getpass'
    self.patch(default.getpass, 'getpass', getpass)

    def raw_input(prompt):
        self.assertEqual(prompt, 'raw_input prompt')
        return 'raw_input'
    self.patch(default, '_input', raw_input)
    d = client.getGenericAnswers(b'Name', b'Instruction', [(b'pass prompt', False), (b'raw_input prompt', True)])
    d.addCallback(self.assertListEqual, ['getpass', 'raw_input'])
    return d