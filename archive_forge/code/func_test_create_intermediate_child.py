from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_create_intermediate_child(self):
    john: TreeNode = TreeNode()
    rose: TreeNode = TreeNode()
    with pytest.raises(KeyError, match='Could not reach'):
        john._set_item(path='Mary/Rose', item=rose, new_nodes_along_path=False)
    john._set_item('Mary/Rose', rose, new_nodes_along_path=True)
    assert 'Mary' in john.children
    mary = john.children['Mary']
    assert isinstance(mary, TreeNode)
    assert mary.children == {'Rose': rose}
    assert rose.parent == mary
    assert rose.parent == mary