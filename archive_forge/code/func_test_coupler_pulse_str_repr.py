import pytest
import sympy
import cirq
import cirq_google.experimental.ops.coupler_pulse as coupler_pulse
def test_coupler_pulse_str_repr():
    gate = coupler_pulse.CouplerPulse(hold_time=cirq.Duration(nanos=10), coupling_mhz=25.0, rise_time=cirq.Duration(nanos=18))
    assert str(gate) == 'CouplerPulse(hold_time=10 ns, coupling_mhz=25.0, rise_time=18 ns, padding_time=2500.0 ps, q0_detune_mhz=0.0, q1_detune_mhz=0.0)'
    assert repr(gate) == 'cirq_google.experimental.ops.coupler_pulse.CouplerPulse(hold_time=cirq.Duration(nanos=10), coupling_mhz=25.0, rise_time=cirq.Duration(nanos=18), padding_time=cirq.Duration(picos=2500.0), q0_detune_mhz=0.0, q1_detune_mhz=0.0)'