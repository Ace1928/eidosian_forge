from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_brokenSelfClosingTag(self) -> None:
    """
        Check that microdom does its best to handle broken self-closing tags
        The important thing is that it doesn't raise an exception.
        """
    self.checkParsed('<body><span /!\n></body>', '<body><span></span></body>')
    self.checkParsed('<span!\n />', '<span></span>')