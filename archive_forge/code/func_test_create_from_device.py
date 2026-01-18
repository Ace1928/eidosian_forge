import pytest
import numpy as np
import google.protobuf.text_format as text_format
import cirq
import cirq_google as cg
import cirq_google.api.v2 as v2
import cirq_google.engine.virtual_engine_factory as factory
def test_create_from_device():
    engine = factory.create_noiseless_virtual_engine_from_device('sycamore', cg.Sycamore)
    _test_processor(engine.get_processor('sycamore'))