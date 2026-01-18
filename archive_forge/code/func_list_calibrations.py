import abc
import datetime
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Union
import duet
import cirq
from cirq_google.api import v2
from cirq_google.cloud import quantum
from cirq_google.engine import calibration
@abc.abstractmethod
def list_calibrations(self, earliest_timestamp: Optional[Union[datetime.datetime, datetime.date, int]]=None, latest_timestamp: Optional[Union[datetime.datetime, datetime.date, int]]=None) -> List[calibration.Calibration]:
    """Retrieve metadata about a specific calibration run.

        Args:
            earliest_timestamp: The earliest timestamp of a calibration
                to return in UTC.
            latest_timestamp: The latest timestamp of a calibration to
                return in UTC.

        Returns:
            The list of calibration data with the most recent first.
        """