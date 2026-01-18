import json
from typing import Callable, cast
from adagio.exceptions import SkippedError
from adagio.instances import (_ConfigVar, _Dependency, _DependencyDict, _Input,
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec
from pytest import raises
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.hash import to_uuid
def test_dependencydict():
    t = MockTaskForVar()
    s = ConfigSpec('a', int, True, False, 1)
    c1 = _ConfigVar(t, s)
    s = ConfigSpec('b', int, True, False, 2)
    c2 = _ConfigVar(t, s)
    d = _DependencyDict(IndexedOrderedDict([('a', c1), ('b', c2)]))
    assert 2 == len(d)
    assert 1 == d['a']
    assert 2 == d['b']
    c2.set(3)
    assert 3 == d['b']
    assert [('a', 1), ('b', 3)] == list(d.items())
    with raises(InvalidOperationError):
        d['c'] = 1
    with raises(InvalidOperationError):
        d['b'] = 1
    with raises(InvalidOperationError):
        d.update(dict())
    assert 3 == d['b']
    assert '3' == d.get_or_throw('b', str)
    assert '3' == d.get('b', 'x')
    assert 0 == d.get('d', 0)
    with raises(KeyError):
        d.get_or_throw('d', str)