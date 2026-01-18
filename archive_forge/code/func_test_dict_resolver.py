from ..resolver import (
def test_dict_resolver():
    resolved = dict_resolver('attr', None, demo_dict, info, **args)
    assert resolved == 'value'