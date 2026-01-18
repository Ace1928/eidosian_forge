from importlib import reload
from typing import Any, Optional
from xml.dom import minidom
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom
def test_findNodesNamed(self):
    doc1 = self.dom.parseString('<doc><foo/><bar/><foo>a</foo></doc>')
    node_list = domhelpers.findNodesNamed(doc1, 'foo')
    actual = len(node_list)
    self.assertEqual(actual, 2)