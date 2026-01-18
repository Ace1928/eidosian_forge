from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testColorAttributes(self) -> None:
    s1 = b'Merry xmas'
    s2 = b'Just kidding'
    self.term.selectGraphicRendition(helper.FOREGROUND + helper.RED, helper.BACKGROUND + helper.GREEN)
    self.term.write(s1 + b'\n')
    self.term.selectGraphicRendition(NORMAL)
    self.term.write(s2 + b'\n')
    for i in range(len(s1)):
        ch = self.term.getCharacter(i, 0)
        self.assertEqual(ch[0], s1[i:i + 1])
        self.assertEqual(ch[1].charset, G0)
        self.assertFalse(ch[1].bold)
        self.assertFalse(ch[1].underline)
        self.assertFalse(ch[1].blink)
        self.assertFalse(ch[1].reverseVideo)
        self.assertEqual(ch[1].foreground, helper.RED)
        self.assertEqual(ch[1].background, helper.GREEN)
    for i in range(len(s2)):
        ch = self.term.getCharacter(i, 1)
        self.assertEqual(ch[0], s2[i:i + 1])
        self.assertEqual(ch[1].charset, G0)
        self.assertFalse(ch[1].bold)
        self.assertFalse(ch[1].underline)
        self.assertFalse(ch[1].blink)
        self.assertFalse(ch[1].reverseVideo)
        self.assertEqual(ch[1].foreground, helper.WHITE)
        self.assertEqual(ch[1].background, helper.BLACK)