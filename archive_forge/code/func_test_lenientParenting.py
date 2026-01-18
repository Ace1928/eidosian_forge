from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_lenientParenting(self) -> None:
    """
        Test that C{parentNode} attributes are set to meaningful values when
        we are parsing HTML that lacks a root node.
        """
    s = '<br/><br/>'
    d = microdom.parseString(s, beExtremelyLenient=1)
    self.assertIdentical(d.documentElement, d.documentElement.firstChild().parentNode)