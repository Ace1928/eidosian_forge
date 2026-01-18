from collections import OrderedDict
import types
from ..pyutil import defaultkeydict, defaultnamedtuple, multi_indexed_cases
def test_multi_indexed_cases():
    for mi, d in multi_indexed_cases([(7, 'abc'), (8, 'ef')]):
        assert type(d) is not dict
        assert isinstance(d, OrderedDict)
        assert isinstance(mi, tuple)
    result = multi_indexed_cases([(97, ['0.5']), (98, ['0.25'])], dict_=dict, apply_return=None, apply_values=float, apply_keys=chr, named_index=True)
    assert isinstance(result, types.GeneratorType)
    (mi, c), = result
    assert mi == (0, 0)
    assert isinstance(mi, tuple)
    assert mi.a == 0
    assert mi.b == 0
    assert mi._asdict() == {'a': 0, 'b': 0}
    assert type(c) is dict
    assert c == {'a': 0.5, 'b': 0.25}