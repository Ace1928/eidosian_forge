import pytest
import sympy
import cirq
import cirq_google.experimental.ops.coupler_pulse as coupler_pulse
@pytest.mark.parametrize('gate, resolver, expected', [(coupler_pulse.CouplerPulse(hold_time=cirq.Duration(nanos=sympy.Symbol('t_ns')), coupling_mhz=10), {'t_ns': 50}, coupler_pulse.CouplerPulse(hold_time=cirq.Duration(nanos=50), coupling_mhz=10)), (coupler_pulse.CouplerPulse(hold_time=cirq.Duration(nanos=50), coupling_mhz=sympy.Symbol('g')), {'g': 10}, coupler_pulse.CouplerPulse(hold_time=cirq.Duration(nanos=50), coupling_mhz=10))])
def test_coupler_pulse_resolution(gate, resolver, expected):
    assert cirq.resolve_parameters(gate, resolver) == expected