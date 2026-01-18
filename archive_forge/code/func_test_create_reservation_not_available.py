import datetime
import pytest
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
@pytest.mark.parametrize('time_slot', (quantum.QuantumTimeSlot(processor_name='test', start_time=Timestamp(seconds=1000000), end_time=Timestamp(seconds=2000000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.OPEN_SWIM), quantum.QuantumTimeSlot(processor_name='test', start_time=Timestamp(seconds=1000000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.OPEN_SWIM), quantum.QuantumTimeSlot(processor_name='test', end_time=Timestamp(seconds=2000000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.OPEN_SWIM)))
def test_create_reservation_not_available(time_slot):
    p = NothingProcessor(processor_id='test', schedule=[time_slot])
    with pytest.raises(ValueError, match='Time slot is not available for reservations'):
        p.create_reservation(start_time=_time(500000), end_time=_time(1500000))