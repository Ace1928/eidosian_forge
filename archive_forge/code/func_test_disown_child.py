from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_disown_child(self):
    mary: TreeNode = TreeNode()
    john: TreeNode = TreeNode(children={'Mary': mary})
    mary.orphan()
    assert mary.parent is None
    assert 'Mary' not in john.children