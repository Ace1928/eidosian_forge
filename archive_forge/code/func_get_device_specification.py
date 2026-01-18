import abc
import datetime
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Union
import duet
import cirq
from cirq_google.api import v2
from cirq_google.cloud import quantum
from cirq_google.engine import calibration
@abc.abstractmethod
def get_device_specification(self) -> Optional[v2.device_pb2.DeviceSpecification]:
    """Returns a device specification proto for use in determining
        information about the device.

        Returns:
            Device specification proto if present.
        """