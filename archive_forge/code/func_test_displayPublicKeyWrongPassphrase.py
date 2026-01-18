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
def test_displayPublicKeyWrongPassphrase(self) -> None:
    """
        L{displayPublicKey} fails with a L{BadKeyError} when trying to decrypt
        an encrypted key with the wrong password.
        """
    filename = self.mktemp()
    FilePath(filename).setContent(privateRSA_openssh_encrypted)
    self.assertRaises(BadKeyError, displayPublicKey, {'filename': filename, 'pass': 'wrong'})