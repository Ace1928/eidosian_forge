from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_cloneDocument(self) -> None:
    s = '<?xml version="1.0"?><!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN""http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"><foo></foo>'
    node = microdom.parseString(s)
    clone = node.cloneNode(deep=1)
    self.failIfEquals(node, clone)
    self.assertEqual(len(node.childNodes), len(clone.childNodes))
    self.assertEqual(s, clone.toxml())
    self.assertTrue(clone.isEqualToDocument(node))
    self.assertTrue(node.isEqualToDocument(clone))