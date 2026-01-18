import datetime
import decimal
from io import BytesIO
import os
import time
from typing import Dict, Union
import uuid
from .const import (
def prepare_local_timestamp_millis(data: Union[datetime.datetime, int], schema: Dict) -> int:
    """Converts datetime.datetime object to int timestamp with milliseconds.

    The local-timestamp-millis logical type represents a timestamp in a local
    timezone, regardless of what specific time zone is considered local, with a
    precision of one millisecond.
    """
    if isinstance(data, datetime.datetime):
        delta = data.replace(tzinfo=datetime.timezone.utc) - epoch
        return (delta.days * 24 * 3600 + delta.seconds) * MLS_PER_SECOND + int(delta.microseconds / 1000)
    else:
        return data