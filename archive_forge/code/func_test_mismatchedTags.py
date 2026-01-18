from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_mismatchedTags(self) -> None:
    for s in ('<test>', '<test> </tset>', '</test>'):
        self.assertRaises(microdom.MismatchedTags, microdom.parseString, s)