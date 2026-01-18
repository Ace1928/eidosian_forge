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
def test_changePassphraseBadKey(self) -> None:
    """
        L{changePassPhrase} exits if the file specified points to an invalid
        key.
        """
    filename = self.mktemp()
    FilePath(filename).setContent(b'foobar')
    error = self.assertRaises(SystemExit, changePassPhrase, {'filename': filename})
    expected = "Could not change passphrase: cannot guess the type of b'foobar'"
    self.assertEqual(expected, str(error))
    self.assertEqual(b'foobar', FilePath(filename).getContent())