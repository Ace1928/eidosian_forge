from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_brokenAttributeName(self) -> None:
    """
        Check that microdom does its best to handle broken attribute names.
        The important thing is that it doesn't raise an exception.
        """
    input = '<body><h1><div al!\n ign="center">Foo</div></h1></body>'
    expected = '<body><h1><div al="True" ign="center">Foo</div></h1></body>'
    self.checkParsed(input, expected)