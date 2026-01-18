from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_isNodeEqualTo(self) -> None:
    """
        L{Node.isEqualToNode} returns C{True} if and only if passed a L{Node}
        with the same children.
        """
    node = microdom.Node(object())
    self.assertTrue(node.isEqualToNode(node))
    another = microdom.Node(object())
    self.assertTrue(node.isEqualToNode(another))
    node.appendChild(microdom.Node(object()))
    self.assertFalse(node.isEqualToNode(another))
    another.appendChild(microdom.Node(object()))
    self.assertTrue(node.isEqualToNode(another))
    node.firstChild().appendChild(microdom.Node(object()))
    self.assertFalse(node.isEqualToNode(another))
    another.firstChild().appendChild(microdom.Node(object()))
    self.assertTrue(node.isEqualToNode(another))