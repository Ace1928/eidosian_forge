from graphql_relay import to_global_id
from ...types import ID, NonNull, ObjectType, String
from ...types.definitions import GrapheneObjectType
from ..node import GlobalID, Node
def test_global_id_defaults_to_required_and_node():
    gid = GlobalID()
    assert isinstance(gid.type, NonNull)
    assert gid.type.of_type == ID
    assert gid.node == Node