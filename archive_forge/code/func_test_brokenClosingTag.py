from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_brokenClosingTag(self) -> None:
    """
        Check that microdom does its best to handle broken closing tags.
        The important thing is that it doesn't raise an exception.
        """
    input = '<body><h1><span>Hello World!</sp!\nan></h1></body>'
    expected = '<body><h1><span>Hello World!</span></h1></body>'
    self.checkParsed(input, expected)
    input = '<body><h1><span>Hello World!</!\nspan></h1></body>'
    self.checkParsed(input, expected)
    input = '<body><h1><span>Hello World!</span!\n></h1></body>'
    self.checkParsed(input, expected)
    input = '<body><h1><span>Hello World!<!\n/span></h1></body>'
    expected = '<body><h1><span>Hello World!<!></!></span></h1></body>'
    self.checkParsed(input, expected)