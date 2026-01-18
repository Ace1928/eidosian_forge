import pytest
import numpy as np
import google.protobuf.text_format as text_format
import cirq
import cirq_google as cg
import cirq_google.api.v2 as v2
import cirq_google.engine.virtual_engine_factory as factory
def test_default_creation():
    engine = factory.create_noiseless_virtual_engine_from_latest_templates()
    _test_processor(engine.get_processor('weber'))
    _test_processor(engine.get_processor('rainbow'))
    for processor_id in ['rainbow', 'weber']:
        processor = engine.get_processor(processor_id)
        device_specification = processor.get_device_specification()
        expected = factory.create_device_spec_from_processor_id(processor_id)
        assert device_specification is not None
        assert device_specification == expected