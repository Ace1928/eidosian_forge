from datetime import (
from typing import Optional
@classmethod
def fromLocalTimeStamp(cls, timeStamp: float) -> 'FixedOffsetTimeZone':
    """
        Create a time zone with a fixed offset corresponding to a time stamp in
        the system's locally configured time zone.
        """
    offset = DateTime.fromtimestamp(timeStamp) - DateTime.fromtimestamp(timeStamp, timezone.utc).replace(tzinfo=None)
    return cls(offset)