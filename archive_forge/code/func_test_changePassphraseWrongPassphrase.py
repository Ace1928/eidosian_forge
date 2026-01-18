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
def test_changePassphraseWrongPassphrase(self) -> None:
    """
        L{changePassPhrase} exits if passed an invalid old passphrase when
        trying to change the passphrase of a private key.
        """
    filename = self.mktemp()
    FilePath(filename).setContent(privateRSA_openssh_encrypted)
    error = self.assertRaises(SystemExit, changePassPhrase, {'filename': filename, 'pass': 'wrong'})
    self.assertEqual('Could not change passphrase: old passphrase error', str(error))
    self.assertEqual(privateRSA_openssh_encrypted, FilePath(filename).getContent())