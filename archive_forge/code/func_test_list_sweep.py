import pytest
import sympy
import cirq
@pytest.mark.parametrize('r_list_factory', [lambda: [{'a': a, 'b': a + 1} for a in (0, 0.5, 1, -10)], lambda: ({'a': a, 'b': a + 1} for a in (0, 0.5, 1, -10)), lambda: ({sympy.Symbol('a'): a, 'b': a + 1} for a in (0, 0.5, 1, -10))])
def test_list_sweep(r_list_factory):
    sweep = cirq.ListSweep(r_list_factory())
    assert sweep.keys == ['a', 'b']
    assert len(sweep) == 4
    assert len(list(sweep)) == 4
    assert list(sweep)[1] == cirq.ParamResolver({'a': 0.5, 'b': 1.5})
    params = list(sweep.param_tuples())
    assert len(params) == 4
    assert params[3] == (('a', -10), ('b', -9))