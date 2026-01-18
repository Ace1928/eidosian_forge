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
def test_host_key_algorithms(self):
    """
        Specify host key algorithms.
        """
    opts = ConchOptions()
    e = self.assertRaises(SystemExit, opts.opt_host_key_algorithms, 'invalid-key')
    self.assertIn('Unknown host key type', e.code)
    opts = ConchOptions()
    opts.opt_host_key_algorithms('ssh-rsa')
    self.assertEqual(opts['host-key-algorithms'], [b'ssh-rsa'])
    opts.opt_host_key_algorithms(b'ssh-dss')
    self.assertEqual(opts['host-key-algorithms'], [b'ssh-dss'])
    opts.opt_host_key_algorithms('ssh-rsa,ssh-dss')
    self.assertEqual(opts['host-key-algorithms'], [b'ssh-rsa', b'ssh-dss'])