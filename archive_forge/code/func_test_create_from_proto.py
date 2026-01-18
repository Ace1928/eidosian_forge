import pytest
import numpy as np
import google.protobuf.text_format as text_format
import cirq
import cirq_google as cg
import cirq_google.api.v2 as v2
import cirq_google.engine.virtual_engine_factory as factory
def test_create_from_proto():
    device_spec = text_format.Merge('\nvalid_qubits: "5_4"\nvalid_gates {\n  phased_xz {}\n}\nvalid_gates {\n  meas {}\n}\n', v2.device_pb2.DeviceSpecification())
    engine = factory.create_noiseless_virtual_engine_from_proto('sycamore', device_spec)
    _test_processor(engine.get_processor('sycamore'))
    assert engine.get_processor('sycamore').get_device_specification() == device_spec
    processor = factory.create_noiseless_virtual_processor_from_proto('sycamore', device_spec)
    _test_processor(processor)
    assert processor.get_device_specification() == device_spec