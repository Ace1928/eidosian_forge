import sys
from typing import NoReturn
from twisted.trial.unittest import TestCase
from twisted.web.template import CDATA, CharRef, Comment, Flattenable, Tag
def test_commentReprPy3(self) -> None:
    """
        L{Comment.__repr__} returns a value which makes it easy to see what's
        in the comment.
        """
    self.assertEqual(repr(Comment('hello there')), "Comment('hello there')")