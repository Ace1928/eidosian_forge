from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testGraphicRendition(self) -> None:
    self.term.selectGraphicRendition(BOLD, UNDERLINE, BLINK, REVERSE_VIDEO)
    self.term.write(b'W')
    self.term.selectGraphicRendition(NORMAL)
    self.term.write(b'X')
    self.term.selectGraphicRendition(BLINK)
    self.term.write(b'Y')
    self.term.selectGraphicRendition(BOLD)
    self.term.write(b'Z')
    ch = self.term.getCharacter(0, 0)
    self.assertEqual(ch[0], b'W')
    self.assertTrue(ch[1].bold)
    self.assertTrue(ch[1].underline)
    self.assertTrue(ch[1].blink)
    self.assertTrue(ch[1].reverseVideo)
    ch = self.term.getCharacter(1, 0)
    self.assertEqual(ch[0], b'X')
    self.assertFalse(ch[1].bold)
    self.assertFalse(ch[1].underline)
    self.assertFalse(ch[1].blink)
    self.assertFalse(ch[1].reverseVideo)
    ch = self.term.getCharacter(2, 0)
    self.assertEqual(ch[0], b'Y')
    self.assertTrue(ch[1].blink)
    self.assertFalse(ch[1].bold)
    self.assertFalse(ch[1].underline)
    self.assertFalse(ch[1].reverseVideo)
    ch = self.term.getCharacter(3, 0)
    self.assertEqual(ch[0], b'Z')
    self.assertTrue(ch[1].blink)
    self.assertTrue(ch[1].bold)
    self.assertFalse(ch[1].underline)
    self.assertFalse(ch[1].reverseVideo)