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
def test_list_time_slots(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    results = [quantum.QuantumTimeSlot(processor_name='potofgold', start_time=Timestamp(seconds=1000020000), end_time=Timestamp(seconds=1000040000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.MAINTENANCE, maintenance_config=quantum.QuantumTimeSlot.MaintenanceConfig(title='Testing', description='Testing some new configuration.')), quantum.QuantumTimeSlot(processor_name='potofgold', start_time=Timestamp(seconds=1000010000), end_time=Timestamp(seconds=1000020000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.RESERVATION, reservation_config=quantum.QuantumTimeSlot.ReservationConfig(project_id='super_secret_quantum'))]
    grpc_client.list_quantum_time_slots.return_value = Pager(results)
    client = EngineClient()
    assert client.list_time_slots('proj', 'processor0') == results