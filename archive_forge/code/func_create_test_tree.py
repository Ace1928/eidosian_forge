from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def create_test_tree() -> tuple[NamedNode, NamedNode]:
    a: NamedNode = NamedNode(name='a')
    b: NamedNode = NamedNode()
    c: NamedNode = NamedNode()
    d: NamedNode = NamedNode()
    e: NamedNode = NamedNode()
    f: NamedNode = NamedNode()
    g: NamedNode = NamedNode()
    h: NamedNode = NamedNode()
    i: NamedNode = NamedNode()
    a.children = {'b': b, 'c': c}
    b.children = {'d': d, 'e': e}
    e.children = {'f': f, 'g': g}
    c.children = {'h': h}
    h.children = {'i': i}
    return (a, f)