import datetime
import decimal
from io import BytesIO
import os
import time
from typing import Dict, Union
import uuid
from .const import (
def prepare_date(data, schema):
    """Converts datetime.date to int timestamp"""
    if isinstance(data, datetime.date):
        return data.toordinal() - DAYS_SHIFT
    elif isinstance(data, str):
        return datetime.date.fromisoformat(data).toordinal() - DAYS_SHIFT
    else:
        return data