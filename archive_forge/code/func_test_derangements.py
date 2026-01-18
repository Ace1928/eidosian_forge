from textwrap import dedent
from itertools import islice, product
from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.sorting import ordered
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.matrices.dense import Matrix
from sympy.combinatorics import RGS_enum, RGS_unrank, Permutation
from sympy.utilities.iterables import (
from sympy.utilities.enumerative import (
from sympy.core.singleton import S
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_derangements():
    assert len(list(generate_derangements(list(range(6))))) == 265
    assert ''.join((''.join(i) for i in generate_derangements('abcde'))) == 'badecbaecdbcaedbcdeabceadbdaecbdeacbdecabeacdbedacbedcacabedcadebcaebdcdaebcdbeacdeabcdebaceabdcebadcedabcedbadabecdaebcdaecbdcaebdcbeadceabdcebadeabcdeacbdebacdebcaeabcdeadbceadcbecabdecbadecdabecdbaedabcedacbedbacedbca'
    assert list(generate_derangements([0, 1, 2, 3])) == [[1, 0, 3, 2], [1, 2, 3, 0], [1, 3, 0, 2], [2, 0, 3, 1], [2, 3, 0, 1], [2, 3, 1, 0], [3, 0, 1, 2], [3, 2, 0, 1], [3, 2, 1, 0]]
    assert list(generate_derangements([0, 1, 2, 2])) == [[2, 2, 0, 1], [2, 2, 1, 0]]
    assert list(generate_derangements('ba')) == [list('ab')]
    D = multiset_derangements
    assert list(D('abb')) == []
    assert [''.join(i) for i in D('ab')] == ['ba']
    assert [''.join(i) for i in D('abc')] == ['bca', 'cab']
    assert [''.join(i) for i in D('aabb')] == ['bbaa']
    assert [''.join(i) for i in D('aabbcccc')] == ['ccccaabb', 'ccccabab', 'ccccabba', 'ccccbaab', 'ccccbaba', 'ccccbbaa']
    assert [''.join(i) for i in D('aabbccc')] == ['cccabba', 'cccabab', 'cccaabb', 'ccacbba', 'ccacbab', 'ccacabb', 'cbccbaa', 'cbccaba', 'cbccaab', 'bcccbaa', 'bcccaba', 'bcccaab']
    assert [''.join(i) for i in D('books')] == ['kbsoo', 'ksboo', 'sbkoo', 'skboo', 'oksbo', 'oskbo', 'okbso', 'obkso', 'oskob', 'oksob', 'osbok', 'obsok']
    assert list(generate_derangements([[3], [2], [2], [1]])) == [[[2], [1], [3], [2]], [[2], [3], [1], [2]]]