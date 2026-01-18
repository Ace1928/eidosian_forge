import datetime
import pytest
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
def test_create_reservation_unbounded():
    time_slot_begin = quantum.QuantumTimeSlot(processor_name='test', end_time=Timestamp(seconds=2000000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED)
    time_slot_end = quantum.QuantumTimeSlot(processor_name='test', start_time=Timestamp(seconds=5000000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED)
    p = NothingProcessor(processor_id='test', schedule=[time_slot_begin, time_slot_end])
    p.create_reservation(start_time=_time(1000000), end_time=_time(3000000))
    p.create_reservation(start_time=_time(4000000), end_time=_time(6000000))
    assert p.get_schedule(from_time=_time(200000), to_time=_time(10000000)) == [quantum.QuantumTimeSlot(processor_name='test', end_time=Timestamp(seconds=1000000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED), quantum.QuantumTimeSlot(processor_name='test', start_time=Timestamp(seconds=1000000), end_time=Timestamp(seconds=3000000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.RESERVATION), quantum.QuantumTimeSlot(processor_name='test', start_time=Timestamp(seconds=4000000), end_time=Timestamp(seconds=6000000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.RESERVATION), quantum.QuantumTimeSlot(processor_name='test', start_time=Timestamp(seconds=6000000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED)]