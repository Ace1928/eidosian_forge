from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_no_time_traveller_loops(self):
    john: TreeNode = TreeNode()
    with pytest.raises(InvalidTreeError, match='cannot be a parent of itself'):
        john._set_parent(john, 'John')
    with pytest.raises(InvalidTreeError, match='cannot be a parent of itself'):
        john.children = {'John': john}
    mary: TreeNode = TreeNode()
    rose: TreeNode = TreeNode()
    mary._set_parent(john, 'Mary')
    rose._set_parent(mary, 'Rose')
    with pytest.raises(InvalidTreeError, match='is already a descendant'):
        john._set_parent(rose, 'John')
    with pytest.raises(InvalidTreeError, match='is already a descendant'):
        rose.children = {'John': john}