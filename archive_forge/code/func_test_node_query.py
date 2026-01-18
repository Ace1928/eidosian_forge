from pytest import mark, raises
from ...types import (
from ...types.scalars import String
from ..mutation import ClientIDMutation
def test_node_query():
    executed = schema.execute('mutation a { say(input: {what:"hello", clientMutationId:"1"}) { phrase } }')
    assert not executed.errors
    assert executed.data == {'say': {'phrase': 'hello'}}