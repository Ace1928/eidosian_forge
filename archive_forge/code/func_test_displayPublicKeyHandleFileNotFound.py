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
def test_displayPublicKeyHandleFileNotFound(self) -> None:
    """
        Ensure FileNotFoundError is handled, whether the user has supplied
        a bad path, or has no key at the default path.
        """
    options = {'filename': '/foo/bar'}
    exc = self.assertRaises(SystemExit, displayPublicKey, options)
    self.assertIn('could not be opened, please specify a file.', exc.args[0])