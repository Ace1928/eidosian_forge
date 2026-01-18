from typing import Any
import cirq
import sympy
def assert_consistent_resolve_parameters(val: Any):
    names = cirq.parameter_names(val)
    symbols = cirq.parameter_symbols(val)
    assert {symbol.name for symbol in symbols} == names
    if not cirq.is_parameterized(val):
        assert not names
        assert not symbols
    else:
        try:
            resolved = cirq.resolve_parameters(val, {name: 0 for name in names})
        except Exception:
            pass
        else:
            assert not cirq.parameter_names(resolved)
            assert not cirq.parameter_symbols(resolved)
        param_dict: cirq.ParamDictType = {name: sympy.Symbol(name + '_CONSISTENCY_TEST') for name in names}
        param_dict.update({sympy.Symbol(name + '_CONSISTENCY_TEST'): 0 for name in names})
        resolver = cirq.ParamResolver(param_dict)
        resolved = cirq.resolve_parameters_once(val, resolver)
        assert cirq.parameter_names(resolved) == set((name + '_CONSISTENCY_TEST' for name in names))