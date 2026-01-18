import abc
import datetime
from typing import Dict, List, Optional, overload, TYPE_CHECKING, Union
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.engine import calibration
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_processor import AbstractProcessor
from cirq_google.engine.abstract_program import AbstractProgram
def get_schedule(self, from_time: Union[None, datetime.datetime, datetime.timedelta]=datetime.timedelta(), to_time: Union[None, datetime.datetime, datetime.timedelta]=datetime.timedelta(weeks=2), time_slot_type: Optional[quantum.QuantumTimeSlot.TimeSlotType]=None) -> List[quantum.QuantumTimeSlot]:
    """Retrieves the schedule for a processor.

        The schedule may be filtered by time.

        Args:
            from_time: Filters the returned schedule to only include entries
                that end no earlier than the given value. Specified either as an
                absolute time (datetime.datetime) or as a time relative to now
                (datetime.timedelta). Defaults to now (a relative time of 0).
                Set to None to omit this filter.
            to_time: Filters the returned schedule to only include entries
                that start no later than the given value. Specified either as an
                absolute time (datetime.datetime) or as a time relative to now
                (datetime.timedelta). Defaults to two weeks from now (a relative
                time of two weeks). Set to None to omit this filter.
            time_slot_type: Filters the returned schedule to only include
                entries with a given type (e.g. maintenance, open swim).
                Defaults to None. Set to None to omit this filter.

        Returns:
            Time slots that fit the criteria.
        """
    time_slots: List[quantum.QuantumTimeSlot] = []
    start_timestamp = _to_timestamp(from_time)
    end_timestamp = _to_timestamp(to_time)
    for slot in self._schedule:
        if start_timestamp and slot.end_time and (slot.end_time.timestamp() < start_timestamp):
            continue
        if end_timestamp and slot.start_time and (slot.start_time.timestamp() > end_timestamp):
            continue
        time_slots.append(slot)
    return time_slots