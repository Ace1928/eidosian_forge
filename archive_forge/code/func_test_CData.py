from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_CData(self) -> None:
    s = '<x><![CDATA[</x>\r\n & foo]]></x>'
    cdata = microdom.parseString(s).documentElement.childNodes[0]
    self.assertTrue(isinstance(cdata, microdom.CDATASection))
    self.assertEqual(cdata.data, '</x>\r\n & foo')
    self.assertEqual(cdata.cloneNode().toxml(), '<![CDATA[</x>\r\n & foo]]>')