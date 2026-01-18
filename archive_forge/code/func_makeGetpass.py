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
def makeGetpass(*passphrases: str) -> Callable[[object], str]:
    """
    Return a callable to patch C{getpass.getpass}.  Yields a passphrase each
    time called. Use case is to provide an old, then new passphrase(s) as if
    requested interactively.

    @param passphrases: The list of passphrases returned, one per each call.

    @return: A callable to patch C{getpass.getpass}.
    """
    passphrasesIter = iter(passphrases)

    def fakeGetpass(_: object) -> str:
        return next(passphrasesIter)
    return fakeGetpass