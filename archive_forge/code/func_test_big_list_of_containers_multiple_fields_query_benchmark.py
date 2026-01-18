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
def test_big_list_of_containers_multiple_fields_query_benchmark(benchmark):

    class Container(ObjectType):
        x = Int()
        y = Int()
        z = Int()
        o = Int()
    big_container_list = [Container(x=x, y=x, z=x, o=x) for x in range(1000)]

    class Query(ObjectType):
        all_containers = List(Container)

        def resolve_all_containers(self, info):
            return big_container_list
    hello_schema = Schema(Query)
    big_list_query = partial(hello_schema.execute, '{ allContainers { x, y, z, o } }')
    result = benchmark(big_list_query)
    assert not result.errors
    assert result.data == {'allContainers': [{'x': c.x, 'y': c.y, 'z': c.z, 'o': c.o} for c in big_container_list]}