import abc
import datetime
from typing import Dict, List, Optional, overload, TYPE_CHECKING, Union
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.engine import calibration
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_processor import AbstractProcessor
from cirq_google.engine.abstract_program import AbstractProgram
def expected_down_time(self) -> 'Optional[datetime.datetime]':
    """Returns the start of the next expected down time of the processor, if
        set."""
    return self._expected_down_time