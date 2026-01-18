from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_eatingWhitespace(self) -> None:
    s = '<hello>\n        </hello>'
    d = microdom.parseString(s)
    self.assertTrue(not d.documentElement.hasChildNodes(), d.documentElement.childNodes)
    self.assertTrue(d.isEqualToDocument(microdom.parseString('<hello></hello>')))