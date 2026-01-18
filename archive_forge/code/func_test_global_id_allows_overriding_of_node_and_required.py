from graphql_relay import to_global_id
from ...types import ID, NonNull, ObjectType, String
from ...types.definitions import GrapheneObjectType
from ..node import GlobalID, Node
def test_global_id_allows_overriding_of_node_and_required():
    gid = GlobalID(node=CustomNode, required=False)
    assert gid.type == ID
    assert gid.node == CustomNode