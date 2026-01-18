import sys
from typing import NoReturn
from twisted.trial.unittest import TestCase
from twisted.web.template import CDATA, CharRef, Comment, Flattenable, Tag
def test_charrefRepr(self) -> None:
    """
        L{CharRef.__repr__} returns a value which makes it easy to see what
        character is referred to.
        """
    snowman = ord('☃')
    self.assertEqual(repr(CharRef(snowman)), 'CharRef(9731)')