from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_bork(self) -> None:
    s = b'<bork><bork><bork>'
    ms = Sux0r()
    ms.connectionMade()
    ms.dataReceived(s)
    self.assertEqual(len(ms.getTagStarts()), 3)