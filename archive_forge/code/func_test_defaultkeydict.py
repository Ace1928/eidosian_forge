from collections import OrderedDict
import types
from ..pyutil import defaultkeydict, defaultnamedtuple, multi_indexed_cases
def test_defaultkeydict():
    d = defaultkeydict(lambda k: k * 2)
    assert d['as'] == 'asas'