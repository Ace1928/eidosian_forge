from ..error import GraphQLError
from ..language.character_classes import is_name_start, is_name_continue
def assert_enum_value_name(name: str) -> str:
    """Uphold the spec rules about naming enum values."""
    assert_name(name)
    if name in {'true', 'false', 'null'}:
        raise GraphQLError(f'Enum values cannot be named: {name}.')
    return name