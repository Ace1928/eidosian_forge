import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring
def test_EyringMassAction():
    args = kB_h_times_exp_dS_R, dH_over_R, c0 = (120000000000.0 / 273.15, 40000.0 / 8, 1)
    ama = MassAction(Eyring(args, ('Sfreq', 'Hact')))
    rxn1 = Reaction({'A': 2, 'B': 1}, {'C': 1}, ama, {'B': 1})
    T_ = 'temperature'

    def ref(v):
        return v.get('Sfreq', 120000000000.0 / 273.15) * v[T_] * math.exp(-v.get('Hact', 40000.0 / 8) / v[T_]) * v['B'] * v['A'] ** 2
    for params in [(11.0, 13.0, 17.0, 311.2), (12, 8, 5, 270)]:
        var = dict(zip(['A', 'B', 'C', T_], params))
        ref_val = ref(var)
        assert abs((ama(var, reaction=rxn1) - ref_val) / ref_val) < 1e-14
    with pytest.raises(ValueError):
        MassAction(Eyring([1, 1, 1, 1, 1]))
    ama2 = MassAction(Eyring4([120000000000.0 / 273, 40000.0 / 8, 1.2, 1000.0], ('Sfreq', 'Hact', 'Sref', 'Href')))
    rxn2 = Reaction({'C': 1}, {'A': 2, 'B': 2}, ama2)
    var2 = {'C': 29, 'temperature': 273}

    def ref2(var):
        return var['C'] * var.get('temperature', 273) * var.get('Sfreq', 120000000000.0 / 273) / var.get('Sref', 1.2) * math.exp((var.get('Href', 1000.0) - var.get('Hact', 5000.0)) / var.get('temperature', 273))
    r2 = ref2(var2)
    assert abs((ama2(var2, reaction=rxn2) - r2) / r2) < 1e-14
    rsys = ReactionSystem([rxn1, rxn2])
    var3 = {'A': 11, 'B': 13, 'C': 17, 'temperature': 298, 'Sfreq': 120000000000.0 / 298}
    rates = rsys.rates(var3)
    rf3 = ref(var3)
    rb3 = ref2(var3)
    ref_rates = {'A': 2 * (rb3 - rf3), 'B': 2 * (rb3 - rf3), 'C': rf3 - rb3}
    for k, v in ref_rates.items():
        assert abs((rates[k] - v) / v) < 1e-14