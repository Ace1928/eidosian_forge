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
def test_enumrepresentationsha256(self) -> None:
    """
        Test for format L{FingerprintFormats.SHA256-BASE64}.
        """
    options = enumrepresentation({'format': 'sha256-base64'})
    self.assertIs(options['format'], FingerprintFormats.SHA256_BASE64)