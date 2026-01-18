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
def test_displayPublicKey(self) -> None:
    """
        L{displayPublicKey} prints out the public key associated with a given
        private key.
        """
    filename = self.mktemp()
    pubKey = Key.fromString(publicRSA_openssh)
    FilePath(filename).setContent(privateRSA_openssh)
    displayPublicKey({'filename': filename})
    displayed = self.stdout.getvalue().strip('\n').encode('ascii')
    self.assertEqual(displayed, pubKey.toString('openssh'))