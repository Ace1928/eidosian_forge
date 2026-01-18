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
def test_changePassphrase(self) -> None:
    """
        L{changePassPhrase} allows a user to change the passphrase of a
        private key interactively.
        """
    oldNewConfirm = makeGetpass('encrypted', 'newpass', 'newpass')
    self.patch(getpass, 'getpass', oldNewConfirm)
    filename = self.mktemp()
    FilePath(filename).setContent(privateRSA_openssh_encrypted)
    changePassPhrase({'filename': filename})
    self.assertEqual(self.stdout.getvalue().strip('\n'), 'Your identification has been saved with the new passphrase.')
    self.assertNotEqual(privateRSA_openssh_encrypted, FilePath(filename).getContent())