from graphql_relay import to_global_id
from ...types import ID, NonNull, ObjectType, String
from ...types.definitions import GrapheneObjectType
from ..node import GlobalID, Node
def test_global_id_defaults_to_info_parent_type():
    my_id = '1'
    gid = GlobalID()
    id_resolver = gid.wrap_resolve(lambda *_: my_id)
    my_global_id = id_resolver(None, Info(User))
    assert my_global_id == to_global_id(User._meta.name, my_id)