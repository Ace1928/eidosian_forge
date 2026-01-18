import json
from functools import partial
from graphql import (
from ..context import Context
from ..dynamic import Dynamic
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import Boolean, Int, String
from ..schema import Schema
from ..structures import List, NonNull
from ..union import Union
def test_default_as_kwarg_to_NonNull():

    class User(ObjectType):
        name = String()
        is_admin = NonNull(Boolean, default_value=False)

    class Query(ObjectType):
        user = Field(User)

        def resolve_user(self, *args, **kwargs):
            return User(name='foo')
    schema = Schema(query=Query)
    expected = {'user': {'name': 'foo', 'isAdmin': False}}
    result = schema.execute('{ user { name isAdmin } }')
    assert not result.errors
    assert result.data == expected