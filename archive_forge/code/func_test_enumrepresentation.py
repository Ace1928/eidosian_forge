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
def test_enumrepresentation(self) -> None:
    """
        L{enumrepresentation} takes a dictionary as input and returns a
        dictionary with its attributes changed to enum representation.
        """
    options = enumrepresentation({'format': 'md5-hex'})
    self.assertIs(options['format'], FingerprintFormats.MD5_HEX)