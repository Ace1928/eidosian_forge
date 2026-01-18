from __future__ import annotations
import getpass
import os
import subprocess
import sys
from io import StringIO
from typing import Callable
from typing_extensions import NoReturn
from twisted.conch.test.keydata import (
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def test_changePassphraseSubtypeV1(self) -> None:
    """
        L{changePassPhrase} can be told to write the new private key file in
        OpenSSH v1 format.
        """
    oldNewConfirm = makeGetpass('encrypted', 'newpass', 'newpass')
    self.patch(getpass, 'getpass', oldNewConfirm)
    filename = self.mktemp()
    FilePath(filename).setContent(privateRSA_openssh_encrypted)
    changePassPhrase({'filename': filename, 'private-key-subtype': 'v1'})
    self.assertEqual(self.stdout.getvalue().strip('\n'), 'Your identification has been saved with the new passphrase.')
    privateKeyContent = FilePath(filename).getContent()
    self.assertNotEqual(privateRSA_openssh_encrypted, privateKeyContent)
    self.assertTrue(privateKeyContent.startswith(b'-----BEGIN OPENSSH PRIVATE KEY-----\n'))