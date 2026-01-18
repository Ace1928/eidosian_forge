import datetime
from unittest import mock
import time
import numpy as np
import pytest
import duet
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge
import cirq
import cirq_google
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.engine import util
from cirq_google.cloud import quantum
from cirq_google.engine.engine import EngineContext
@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_create_engine(client):
    with pytest.raises(ValueError, match='provide context or proto_version, service_args and verbose'):
        cg.Engine('proj', proto_version=cg.engine.engine.ProtoVersion.V2, service_args={'args': 'test'}, verbose=True, context=mock.Mock())
    assert cg.Engine('proj', proto_version=cg.engine.engine.ProtoVersion.V2, service_args={'args': 'test'}, verbose=True).context.proto_version == cg.engine.engine.ProtoVersion.V2
    client.assert_called_with({'args': 'test'}, True)