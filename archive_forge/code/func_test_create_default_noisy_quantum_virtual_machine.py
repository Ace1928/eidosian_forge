import pytest
import numpy as np
import google.protobuf.text_format as text_format
import cirq
import cirq_google as cg
import cirq_google.api.v2 as v2
import cirq_google.engine.virtual_engine_factory as factory
def test_create_default_noisy_quantum_virtual_machine():
    for processor_id in ['rainbow', 'weber']:
        engine = factory.create_default_noisy_quantum_virtual_machine(processor_id=processor_id, simulator_class=cirq.Simulator)
        processor = engine.get_processor(processor_id)
        bad_qubit = cirq.GridQubit(10, 10)
        circuit = cirq.Circuit(cirq.X(bad_qubit), cirq.measure(bad_qubit))
        with pytest.raises(ValueError, match='Qubit not on device'):
            _ = processor.run(circuit, repetitions=100)
        good_qubit = cirq.GridQubit(5, 4)
        circuit = cirq.Circuit(cirq.H(good_qubit), cirq.measure(good_qubit))
        with pytest.raises(ValueError, match='.* contains a gate which is not supported.'):
            _ = processor.run(circuit, repetitions=100)
        device_specification = processor.get_device_specification()
        expected = factory.create_device_spec_from_processor_id(processor_id)
        assert device_specification is not None
        assert device_specification == expected