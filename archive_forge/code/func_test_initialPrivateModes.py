from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def test_initialPrivateModes(self) -> None:
    """
        Verify that only DEC Auto Wrap Mode (DECAWM) and DEC Text Cursor Enable
        Mode (DECTCEM) are initially in the Set Mode (SM) state.
        """
    self.assertEqual({privateModes.AUTO_WRAP: True, privateModes.CURSOR_MODE: True}, self.term.privateModes)