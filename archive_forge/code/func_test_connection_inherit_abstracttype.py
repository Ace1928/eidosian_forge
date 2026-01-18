import re
from pytest import raises
from ...types import Argument, Field, Int, List, NonNull, ObjectType, Schema, String
from ..connection import (
from ..node import Node
def test_connection_inherit_abstracttype():

    class BaseConnection:
        extra = String()

    class MyObjectConnection(BaseConnection, Connection):

        class Meta:
            node = MyObject
    assert MyObjectConnection._meta.name == 'MyObjectConnection'
    fields = MyObjectConnection._meta.fields
    assert list(fields) == ['page_info', 'edges', 'extra']