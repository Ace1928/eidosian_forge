import re
from pytest import raises
from ...types import Argument, Field, Int, List, NonNull, ObjectType, Schema, String
from ..connection import (
from ..node import Node
def test_connectionfield_custom_args():

    class MyObjectConnection(Connection):

        class Meta:
            node = MyObject
    field = ConnectionField(MyObjectConnection, before=String(required=True), extra=String())
    assert field.args == {'before': Argument(NonNull(String)), 'after': Argument(String), 'first': Argument(Int), 'last': Argument(Int), 'extra': Argument(String)}