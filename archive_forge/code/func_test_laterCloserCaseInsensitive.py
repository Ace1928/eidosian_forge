from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_laterCloserCaseInsensitive(self) -> None:
    s = '<DL><p><DT>foo<DD>bar</DL>'
    d = microdom.parseString(s, beExtremelyLenient=1)
    expected = '<dl><p></p><dt>foo</dt><dd>bar</dd></dl>'
    actual = d.documentElement.toxml()
    self.assertEqual(expected, actual)