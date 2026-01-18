from ..resolver import (
def test_dict_resolver_default_value():
    resolved = dict_resolver('attr2', 'default', demo_dict, info, **args)
    assert resolved == 'default'