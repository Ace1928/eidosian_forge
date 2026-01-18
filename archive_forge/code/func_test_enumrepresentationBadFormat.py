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
def test_enumrepresentationBadFormat(self) -> None:
    """
        Test for unsupported fingerprint format
        """
    with self.assertRaises(BadFingerPrintFormat) as em:
        enumrepresentation({'format': 'sha-base64'})
    self.assertEqual('Unsupported fingerprint format: sha-base64', em.exception.args[0])