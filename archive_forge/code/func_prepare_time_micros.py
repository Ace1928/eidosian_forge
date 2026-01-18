import datetime
import decimal
from io import BytesIO
import os
import time
from typing import Dict, Union
import uuid
from .const import (
def prepare_time_micros(data, schema):
    """Convert datetime.time to int timestamp with microseconds"""
    if isinstance(data, datetime.time):
        return int(data.hour * MCS_PER_HOUR + data.minute * MCS_PER_MINUTE + data.second * MCS_PER_SECOND + data.microsecond)
    else:
        return data