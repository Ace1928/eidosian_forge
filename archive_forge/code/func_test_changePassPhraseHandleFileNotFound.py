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
def test_changePassPhraseHandleFileNotFound(self) -> None:
    """
        Ensure FileNotFoundError is handled for an invalid filename.
        """
    options = {'filename': '/foo/bar'}
    exc = self.assertRaises(SystemExit, changePassPhrase, options)
    self.assertIn('could not be opened, please specify a file.', exc.args[0])