import pytest
import numpy as np
import google.protobuf.text_format as text_format
import cirq
import cirq_google as cg
import cirq_google.api.v2 as v2
import cirq_google.engine.virtual_engine_factory as factory
@pytest.mark.parametrize('processor_id', ['rainbow', 'weber'])
def test_median_device_expected_fields(processor_id):
    cal = factory.load_median_device_calibration(processor_id)
    expected_fields = {'single_qubit_idle_t1_micros', 'single_qubit_rb_incoherent_error_per_gate', 'single_qubit_rb_pauli_error_per_gate', 'two_qubit_parallel_sycamore_gate_xeb_pauli_error_per_cycle', 'two_qubit_parallel_sqrt_iswap_gate_xeb_pauli_error_per_cycle', 'single_qubit_p00_error', 'single_qubit_p11_error', 'two_qubit_parallel_sycamore_gate_xeb_entangler_theta_error_per_cycle', 'two_qubit_parallel_sycamore_gate_xeb_entangler_phi_error_per_cycle', 'two_qubit_parallel_sqrt_iswap_gate_xeb_entangler_theta_error_per_cycle', 'two_qubit_parallel_sqrt_iswap_gate_xeb_entangler_phi_error_per_cycle'}
    assert expected_fields.issubset(cal.keys())