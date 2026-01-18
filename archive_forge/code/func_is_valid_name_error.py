from typing import Optional
from ..type.assert_name import assert_name
from ..error import GraphQLError
def is_valid_name_error(name: str) -> Optional[GraphQLError]:
    """Return an Error if a name is invalid.

    .. deprecated:: 3.2
       Please use ``assert_name`` instead. Will be removed in v3.3.
    """
    if not isinstance(name, str):
        raise TypeError('Expected name to be a string.')
    if name.startswith('__'):
        return GraphQLError(f"Name {name!r} must not begin with '__', which is reserved by GraphQL introspection.")
    try:
        assert_name(name)
    except GraphQLError as error:
        return error
    return None