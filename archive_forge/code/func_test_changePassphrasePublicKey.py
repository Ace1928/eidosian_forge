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
def test_changePassphrasePublicKey(self) -> None:
    """
        L{changePassPhrase} exits when trying to change the passphrase on a
        public key, and doesn't change the file.
        """
    filename = self.mktemp()
    FilePath(filename).setContent(publicRSA_openssh)
    error = self.assertRaises(SystemExit, changePassPhrase, {'filename': filename, 'newpass': 'pass'})
    self.assertEqual('Could not change passphrase: key not encrypted', str(error))
    self.assertEqual(publicRSA_openssh, FilePath(filename).getContent())