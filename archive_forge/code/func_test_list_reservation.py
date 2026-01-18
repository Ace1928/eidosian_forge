import asyncio
import datetime
import os
from unittest import mock
import duet
import pytest
from google.api_core import exceptions
from google.protobuf import any_pb2
from google.protobuf.field_mask_pb2 import FieldMask
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.engine.engine_client import EngineClient, EngineException
import cirq_google.engine.stream_manager as engine_stream_manager
from cirq_google.cloud import quantum
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_list_reservation(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    name = 'projects/proj/processors/processor0/reservations/papar-party-44'
    results = [quantum.QuantumReservation(name=name, start_time=Timestamp(seconds=1000000000), end_time=Timestamp(seconds=1000002000), whitelisted_users=['jeff@google.com']), quantum.QuantumReservation(name=name, start_time=Timestamp(seconds=1200000000), end_time=Timestamp(seconds=1200002000), whitelisted_users=['dstrain@google.com'])]
    grpc_client.list_quantum_reservations.return_value = Pager(results)
    client = EngineClient()
    assert client.list_reservations('proj', 'processor0') == results