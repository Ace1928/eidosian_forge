import datetime
import pytest
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
def test_create_reservation_add_at_end():
    time_slot = quantum.QuantumTimeSlot(processor_name='test', start_time=Timestamp(seconds=1000000), end_time=Timestamp(seconds=2000000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED)
    p = NothingProcessor(processor_id='test', schedule=[time_slot])
    p.create_reservation(start_time=_time(2500000), end_time=_time(3500000))
    assert p.get_schedule(from_time=_time(500000), to_time=_time(2500000)) == [quantum.QuantumTimeSlot(processor_name='test', start_time=Timestamp(seconds=1000000), end_time=Timestamp(seconds=2000000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED), quantum.QuantumTimeSlot(processor_name='test', start_time=Timestamp(seconds=2500000), end_time=Timestamp(seconds=3500000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.RESERVATION)]