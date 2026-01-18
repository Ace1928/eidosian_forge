import pytest
import numpy as np
import google.protobuf.text_format as text_format
import cirq
import cirq_google as cg
import cirq_google.api.v2 as v2
import cirq_google.engine.virtual_engine_factory as factory
def test_create_device_from_processor_id():
    device = factory.create_device_from_processor_id('rainbow')
    assert device is not None
    with pytest.raises(ValueError, match='no such processor is defined'):
        _ = factory.create_device_from_processor_id('bad_processor')