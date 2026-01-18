from itertools import permutations
from sympy.core.expr import unchanged
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.core.singleton import S
from sympy.combinatorics.permutations import \
from sympy.printing import sstr, srepr, pretty, latex
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_Permutation():
    raises(ValueError, lambda: Permutation([1]))
    p = Permutation([0, 1, 2, 3])
    assert [p(i) for i in range(p.size)] == list(p)
    assert p(list(range(p.size))) == list(p)
    assert list(p(1, 2)) == [0, 2, 1, 3]
    raises(TypeError, lambda: p(-1))
    raises(TypeError, lambda: p(5))
    assert list(p) == list(range(4))
    assert Permutation(size=4) == Permutation(3)
    assert Permutation(Permutation(3), size=5) == Permutation(4)
    assert Permutation([[1, 2]], size=4) == Permutation([[1, 2], [0], [3]])
    assert Permutation.random(2) in (Permutation([1, 0]), Permutation([0, 1]))
    p = Permutation([2, 5, 1, 6, 3, 0, 4])
    q = Permutation([[1], [0, 3, 5, 6, 2, 4]])
    assert len({p, p}) == 1
    r = Permutation([1, 3, 2, 0, 4, 6, 5])
    ans = Permutation(_af_rmuln(*[w.array_form for w in (p, q, r)])).array_form
    assert rmul(p, q, r).array_form == ans
    for a, b, c in permutations((p, q, r)):
        if (a, b, c) == (p, q, r):
            continue
        assert rmul(a, b, c).array_form != ans
    assert p.support() == list(range(7))
    assert q.support() == [0, 2, 3, 4, 5, 6]
    assert Permutation(p.cyclic_form).array_form == p.array_form
    assert p.cardinality == 5040
    assert q.cardinality == 5040
    assert q.cycles == 2
    assert rmul(q, p) == Permutation([4, 6, 1, 2, 5, 3, 0])
    assert rmul(p, q) == Permutation([6, 5, 3, 0, 2, 4, 1])
    assert _af_rmul(p.array_form, q.array_form) == [6, 5, 3, 0, 2, 4, 1]
    assert rmul(Permutation([[1, 2, 3], [0, 4]]), Permutation([[1, 2, 4], [0], [3]])).cyclic_form == [[0, 4, 2], [1, 3]]
    assert q.array_form == [3, 1, 4, 5, 0, 6, 2]
    assert q.cyclic_form == [[0, 3, 5, 6, 2, 4]]
    assert q.full_cyclic_form == [[0, 3, 5, 6, 2, 4], [1]]
    assert p.cyclic_form == [[0, 2, 1, 5], [3, 6, 4]]
    t = p.transpositions()
    assert t == [(0, 5), (0, 1), (0, 2), (3, 4), (3, 6)]
    assert Permutation.rmul(*[Permutation(Cycle(*ti)) for ti in t])
    assert Permutation([1, 0]).transpositions() == [(0, 1)]
    assert p ** 13 == p
    assert q ** 0 == Permutation(list(range(q.size)))
    assert q ** (-2) == ~q ** 2
    assert q ** 2 == Permutation([5, 1, 0, 6, 3, 2, 4])
    assert q ** 3 == q ** 2 * q
    assert q ** 4 == q ** 2 * q ** 2
    a = Permutation(1, 3)
    b = Permutation(2, 0, 3)
    I = Permutation(3)
    assert ~a == a ** (-1)
    assert a * ~a == I
    assert a * b ** (-1) == a * ~b
    ans = Permutation(0, 5, 3, 1, 6)(2, 4)
    assert (p + q.rank()).rank() == ans.rank()
    assert (p + q.rank())._rank == ans.rank()
    assert (q + p.rank()).rank() == ans.rank()
    raises(TypeError, lambda: p + Permutation(list(range(10))))
    assert (p - q.rank()).rank() == Permutation(0, 6, 3, 1, 2, 5, 4).rank()
    assert p.rank() - q.rank() < 0
    assert (q - p.rank()).rank() == Permutation(1, 4, 6, 2)(3, 5).rank()
    assert p * q == Permutation(_af_rmuln(*[list(w) for w in (q, p)]))
    assert p * Permutation([]) == p
    assert Permutation([]) * p == p
    assert p * Permutation([[0, 1]]) == Permutation([2, 5, 0, 6, 3, 1, 4])
    assert Permutation([[0, 1]]) * p == Permutation([5, 2, 1, 6, 3, 0, 4])
    pq = p ^ q
    assert pq == Permutation([5, 6, 0, 4, 1, 2, 3])
    assert pq == rmul(q, p, ~q)
    qp = q ^ p
    assert qp == Permutation([4, 3, 6, 2, 1, 5, 0])
    assert qp == rmul(p, q, ~p)
    raises(ValueError, lambda: p ^ Permutation([]))
    assert p.commutator(q) == Permutation(0, 1, 3, 4, 6, 5, 2)
    assert q.commutator(p) == Permutation(0, 2, 5, 6, 4, 3, 1)
    assert p.commutator(q) == ~q.commutator(p)
    raises(ValueError, lambda: p.commutator(Permutation([])))
    assert len(p.atoms()) == 7
    assert q.atoms() == {0, 1, 2, 3, 4, 5, 6}
    assert p.inversion_vector() == [2, 4, 1, 3, 1, 0]
    assert q.inversion_vector() == [3, 1, 2, 2, 0, 1]
    assert Permutation.from_inversion_vector(p.inversion_vector()) == p
    assert Permutation.from_inversion_vector(q.inversion_vector()).array_form == q.array_form
    raises(ValueError, lambda: Permutation.from_inversion_vector([0, 2]))
    assert Permutation(list(range(500, -1, -1))).inversions() == 125250
    s = Permutation([0, 4, 1, 3, 2])
    assert s.parity() == 0
    _ = s.cyclic_form
    assert len(s._cyclic_form) != s.size and s.parity() == 0
    assert not s.is_odd
    assert s.is_even
    assert Permutation([0, 1, 4, 3, 2]).parity() == 1
    assert _af_parity([0, 4, 1, 3, 2]) == 0
    assert _af_parity([0, 1, 4, 3, 2]) == 1
    s = Permutation([0])
    assert s.is_Singleton
    assert Permutation([]).is_Empty
    r = Permutation([3, 2, 1, 0])
    assert (r ** 2).is_Identity
    assert rmul(~p, p).is_Identity
    assert (~p) ** 13 == Permutation([5, 2, 0, 4, 6, 1, 3])
    assert ~(r ** 2).is_Identity
    assert p.max() == 6
    assert p.min() == 0
    q = Permutation([[6], [5], [0, 1, 2, 3, 4]])
    assert q.max() == 4
    assert q.min() == 0
    p = Permutation([1, 5, 2, 0, 3, 6, 4])
    q = Permutation([[1, 2, 3, 5, 6], [0, 4]])
    assert p.ascents() == [0, 3, 4]
    assert q.ascents() == [1, 2, 4]
    assert r.ascents() == []
    assert p.descents() == [1, 2, 5]
    assert q.descents() == [0, 3, 5]
    assert Permutation(r.descents()).is_Identity
    assert p.inversions() == 7
    big = list(p) + list(range(p.max() + 1, p.max() + 130))
    assert Permutation(big).inversions() == 7
    assert p.signature() == -1
    assert q.inversions() == 11
    assert q.signature() == -1
    assert rmul(p, ~p).inversions() == 0
    assert rmul(p, ~p).signature() == 1
    assert p.order() == 6
    assert q.order() == 10
    assert (p ** p.order()).is_Identity
    assert p.length() == 6
    assert q.length() == 7
    assert r.length() == 4
    assert p.runs() == [[1, 5], [2], [0, 3, 6], [4]]
    assert q.runs() == [[4], [2, 3, 5], [0, 6], [1]]
    assert r.runs() == [[3], [2], [1], [0]]
    assert p.index() == 8
    assert q.index() == 8
    assert r.index() == 3
    assert p.get_precedence_distance(q) == q.get_precedence_distance(p)
    assert p.get_adjacency_distance(q) == p.get_adjacency_distance(q)
    assert p.get_positional_distance(q) == p.get_positional_distance(q)
    p = Permutation([0, 1, 2, 3])
    q = Permutation([3, 2, 1, 0])
    assert p.get_precedence_distance(q) == 6
    assert p.get_adjacency_distance(q) == 3
    assert p.get_positional_distance(q) == 8
    p = Permutation([0, 3, 1, 2, 4])
    q = Permutation.josephus(4, 5, 2)
    assert p.get_adjacency_distance(q) == 3
    raises(ValueError, lambda: p.get_adjacency_distance(Permutation([])))
    raises(ValueError, lambda: p.get_positional_distance(Permutation([])))
    raises(ValueError, lambda: p.get_precedence_distance(Permutation([])))
    a = [Permutation.unrank_nonlex(4, i) for i in range(5)]
    iden = Permutation([0, 1, 2, 3])
    for i in range(5):
        for j in range(i + 1, 5):
            assert a[i].commutes_with(a[j]) == (rmul(a[i], a[j]) == rmul(a[j], a[i]))
            if a[i].commutes_with(a[j]):
                assert a[i].commutator(a[j]) == iden
                assert a[j].commutator(a[i]) == iden
    a = Permutation(3)
    b = Permutation(0, 6, 3)(1, 2)
    assert a.cycle_structure == {1: 4}
    assert b.cycle_structure == {2: 1, 3: 1, 1: 2}
    raises(ValueError, lambda: Permutation(3, size=3))
    raises(ValueError, lambda: Permutation([1, 2, 0, 3], size=3))