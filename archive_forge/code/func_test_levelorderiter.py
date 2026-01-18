from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_levelorderiter(self):
    root, _ = create_test_tree()
    result: list[str | None] = [node.name for node in cast(Iterator[NamedNode], LevelOrderIter(root))]
    expected = ['a', 'b', 'c', 'd', 'e', 'h', 'f', 'g', 'i']
    assert result == expected