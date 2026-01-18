from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_leaves(self):
    tree, _ = create_test_tree()
    leaves = tree.leaves
    expected = ['d', 'f', 'g', 'i']
    for node, expected_name in zip(leaves, expected):
        assert node.name == expected_name