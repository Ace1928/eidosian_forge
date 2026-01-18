from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_laterCloserDL(self) -> None:
    s = '<dl><dt>word<dd>definition<dt>word<dt>word<dd>definition<dd>definition</dl>'
    expected = '<dl><dt>word</dt><dd>definition</dd><dt>word</dt><dt>word</dt><dd>definition</dd><dd>definition</dd></dl>'
    d = microdom.parseString(s, beExtremelyLenient=1)
    actual = d.documentElement.toxml()
    self.assertEqual(expected, actual)