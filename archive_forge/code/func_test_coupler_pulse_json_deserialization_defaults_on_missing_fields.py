import pytest
import sympy
import cirq
import cirq_google.experimental.ops.coupler_pulse as coupler_pulse
def test_coupler_pulse_json_deserialization_defaults_on_missing_fields():
    gate = coupler_pulse.CouplerPulse(hold_time=cirq.Duration(nanos=10), coupling_mhz=25.0, rise_time=cirq.Duration(nanos=18))
    json_text = '{\n       "cirq_type": "CouplerPulse",\n       "hold_time": {\n         "cirq_type": "Duration",\n         "picos": 10000\n       },\n       "coupling_mhz": 25.0,\n       "rise_time": {\n         "cirq_type": "Duration",\n         "picos": 18000\n       },\n       "padding_time": {\n         "cirq_type": "Duration",\n         "picos": 2500.0\n       }\n    }'
    deserialized = cirq.read_json(json_text=json_text)
    assert deserialized == gate
    assert deserialized.q0_detune_mhz == 0.0