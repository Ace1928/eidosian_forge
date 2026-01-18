import datetime
import decimal
from io import BytesIO
import os
import time
from typing import Dict, Union
import uuid
from .const import (
def prepare_time_millis(data, schema):
    """Convert datetime.time to int timestamp with milliseconds"""
    if isinstance(data, datetime.time):
        return int(data.hour * MLS_PER_HOUR + data.minute * MLS_PER_MINUTE + data.second * MLS_PER_SECOND + int(data.microsecond / 1000))
    else:
        return data