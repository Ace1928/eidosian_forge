from __future__ import annotations
from typing import Callable
from typing_extensions import NoReturn, Protocol
from twisted.python import randbytes
from twisted.trial import unittest
def errorFactory(self, nbytes: object) -> NoReturn:
    """
        A factory raising an error when a source is not available.
        """
    raise randbytes.SourceNotAvailable()