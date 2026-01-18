import re
from pytest import raises
from ...types import Argument, Field, Int, List, NonNull, ObjectType, Schema, String
from ..connection import (
from ..node import Node
def test_edge_with_bases():

    class BaseEdge:
        extra = String()

    class MyObjectConnection(Connection):

        class Meta:
            node = MyObject

        class Edge(BaseEdge):
            other = String()
    Edge = MyObjectConnection.Edge
    assert Edge._meta.name == 'MyObjectEdge'
    edge_fields = Edge._meta.fields
    assert list(edge_fields) == ['node', 'cursor', 'extra', 'other']
    assert isinstance(edge_fields['node'], Field)
    assert edge_fields['node'].type == MyObject
    assert isinstance(edge_fields['other'], Field)
    assert edge_fields['other'].type == String