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
def test_create_reservation(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    start = datetime.datetime.fromtimestamp(1000000000)
    end = datetime.datetime.fromtimestamp(1000003600)
    users = ['jeff@google.com']
    result = quantum.QuantumReservation(name='projects/proj/processors/processor0/reservations/papar-party-44', start_time=Timestamp(seconds=1000000000), end_time=Timestamp(seconds=1000003600), whitelisted_users=users)
    grpc_client.create_quantum_reservation.return_value = result
    client = EngineClient()
    assert client.create_reservation('proj', 'processor0', start, end, users) == result
    assert grpc_client.create_quantum_reservation.call_count == 1
    result.name = ''
    grpc_client.create_quantum_reservation.assert_called_with(quantum.CreateQuantumReservationRequest(parent='projects/proj/processors/processor0', quantum_reservation=result))