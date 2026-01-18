from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_awfulTagSoup(self) -> None:
    s = '\n        <html>\n        <head><title> I send you this message to have your advice!!!!</titl e\n        </headd>\n\n        <body bgcolor alink hlink vlink>\n\n        <h1><BLINK>SALE</blINK> TWENTY MILLION EMAILS & FUR COAT NOW\n        FREE WITH `ENLARGER\'</h1>\n\n        YES THIS WONDERFUL AWFER IS NOW HERER!!!\n\n        <script LANGUAGE="javascript">\nfunction give_answers() {\nif (score < 70) {\nalert("I hate you");\n}}\n        </script><a href=/foo.com/lalal name=foo>lalal</a>\n        </body>\n        </HTML>\n        '
    d = microdom.parseString(s, beExtremelyLenient=1)
    l = domhelpers.findNodesNamed(d.documentElement, 'blink')
    self.assertEqual(len(l), 1)