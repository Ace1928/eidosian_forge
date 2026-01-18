from typing import Sequence
import numpy as np
import pytest
import cirq
def test_assert_act_on_clifford_tableau_effect_matches_unitary():
    cirq.testing.assert_all_implemented_act_on_effects_match_unitary(GoodGate())
    cirq.testing.assert_all_implemented_act_on_effects_match_unitary(GoodGate().on(cirq.LineQubit(1)))
    with pytest.raises(AssertionError, match='act_on clifford tableau is not consistent with final_state_vector simulation.'):
        cirq.testing.assert_all_implemented_act_on_effects_match_unitary(BadGate())
    cirq.testing.assert_all_implemented_act_on_effects_match_unitary(UnimplementedGate())
    with pytest.raises(AssertionError, match='Could not assert if any act_on methods were implemented'):
        cirq.testing.assert_all_implemented_act_on_effects_match_unitary(UnimplementedGate(), assert_tableau_implemented=True)
    with pytest.raises(AssertionError, match='Could not assert if any act_on methods were implemented'):
        cirq.testing.assert_all_implemented_act_on_effects_match_unitary(UnimplementedGate(), assert_ch_form_implemented=True)
    cirq.testing.assert_all_implemented_act_on_effects_match_unitary(UnimplementedUnitaryGate())
    with pytest.raises(AssertionError, match='Failed to generate final tableau'):
        cirq.testing.assert_all_implemented_act_on_effects_match_unitary(UnimplementedUnitaryGate(), assert_tableau_implemented=True)
    with pytest.raises(AssertionError, match='Failed to generate final stabilizer state'):
        cirq.testing.assert_all_implemented_act_on_effects_match_unitary(UnimplementedUnitaryGate(), assert_ch_form_implemented=True)