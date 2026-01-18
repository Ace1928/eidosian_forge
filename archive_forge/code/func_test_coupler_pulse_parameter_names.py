import pytest
import sympy
import cirq
import cirq_google.experimental.ops.coupler_pulse as coupler_pulse
@pytest.mark.parametrize('gate, param_names', [(coupler_pulse.CouplerPulse(hold_time=cirq.Duration(nanos=sympy.Symbol('t_ns')), coupling_mhz=10), {'t_ns'}), (coupler_pulse.CouplerPulse(hold_time=cirq.Duration(nanos=50), coupling_mhz=sympy.Symbol('g')), {'g'}), (coupler_pulse.CouplerPulse(hold_time=cirq.Duration(nanos=sympy.Symbol('t_ns')), coupling_mhz=sympy.Symbol('g')), {'g', 't_ns'})])
def test_coupler_pulse_parameter_names(gate, param_names):
    assert cirq.parameter_names(gate) == param_names