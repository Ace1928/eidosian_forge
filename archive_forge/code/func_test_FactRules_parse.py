from sympy.core.facts import (deduce_alpha_implications,
from sympy.core.logic import And, Not
from sympy.testing.pytest import raises
def test_FactRules_parse():
    f = FactRules('a -> b')
    assert f.prereq == {'b': {'a'}, 'a': {'b'}}
    f = FactRules('a -> !b')
    assert f.prereq == {'b': {'a'}, 'a': {'b'}}
    f = FactRules('!a -> b')
    assert f.prereq == {'b': {'a'}, 'a': {'b'}}
    f = FactRules('!a -> !b')
    assert f.prereq == {'b': {'a'}, 'a': {'b'}}
    f = FactRules('!z == nz')
    assert f.prereq == {'z': {'nz'}, 'nz': {'z'}}