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
def test_changePassphraseEmptyStringError(self) -> None:
    """
        L{changePassPhrase} doesn't modify the key file if C{toString} returns
        an empty string.
        """
    filename = self.mktemp()
    FilePath(filename).setContent(privateRSA_openssh)

    def toString(*args: object, **kwargs: object) -> str:
        return ''
    self.patch(Key, 'toString', toString)
    error = self.assertRaises(SystemExit, changePassPhrase, {'filename': filename, 'newpass': 'newencrypt'})
    expected = "Could not change passphrase: cannot guess the type of b''"
    self.assertEqual(expected, str(error))
    self.assertEqual(privateRSA_openssh, FilePath(filename).getContent())