import datetime
import decimal
from io import BytesIO
import os
import time
from typing import Dict, Union
import uuid
from .const import (
def prepare_local_timestamp_micros(data: Union[datetime.datetime, int], schema: Dict) -> int:
    """Converts datetime.datetime to int timestamp with microseconds

    The local-timestamp-micros logical type represents a timestamp in a local
    timezone, regardless of what specific time zone is considered local, with a
    precision of one microsecond.
    """
    if isinstance(data, datetime.datetime):
        delta = data.replace(tzinfo=datetime.timezone.utc) - epoch
        return (delta.days * 24 * 3600 + delta.seconds) * MCS_PER_SECOND + delta.microseconds
    else:
        return data