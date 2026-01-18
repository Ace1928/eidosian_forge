import abc
import datetime
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Union
import duet
import cirq
from cirq_google.api import v2
from cirq_google.cloud import quantum
from cirq_google.engine import calibration
@abc.abstractmethod
def get_current_calibration(self) -> Optional[calibration.Calibration]:
    """Returns metadata about the current calibration for a processor.

        Returns:
            The calibration data or None if there is no current calibration.
        """