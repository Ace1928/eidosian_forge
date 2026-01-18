from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_child_already_exists(self):
    mary: TreeNode = TreeNode()
    john: TreeNode = TreeNode(children={'Mary': mary})
    mary_2: TreeNode = TreeNode()
    with pytest.raises(KeyError):
        john._set_item('Mary', mary_2, allow_overwrite=False)