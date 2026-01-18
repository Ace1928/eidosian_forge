from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
def getTime() -> float:
    """
        Get time when delayed call will happen.

        @return: time in seconds since epoch (a float).
        """