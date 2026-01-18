from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
def removeEvent(event: object) -> None:
    """
        Remove an event.

        @param event: a Win32 event object added using L{IReactorWin32Events.addEvent}

        @return: None
        """