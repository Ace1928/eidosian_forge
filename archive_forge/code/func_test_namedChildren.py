from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_namedChildren(self) -> None:
    tests = {"<foo><bar /><bar unf='1' /><bar>asdfadsf</bar><bam/></foo>": 3, '<foo>asdf</foo>': 0, '<foo><bar><bar></bar></bar></foo>': 1}
    for t in tests.keys():
        node = microdom.parseString(t).documentElement
        result = domhelpers.namedChildren(node, 'bar')
        self.assertEqual(len(result), tests[t])
        if result:
            self.assertTrue(hasattr(result[0], 'tagName'))