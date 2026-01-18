import abc
import datetime
from typing import Dict, List, Optional, overload, TYPE_CHECKING, Union
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.engine import calibration
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_processor import AbstractProcessor
from cirq_google.engine.abstract_program import AbstractProgram
def remove_reservation(self, reservation_id: str) -> None:
    """Removes a reservation on this processor."""
    if reservation_id in self._reservations:
        del self._reservations[reservation_id]