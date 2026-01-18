from ..resolver import (
def test_get_default_resolver_is_attr_resolver():
    assert get_default_resolver() == dict_or_attr_resolver