import uuid
from datetime import datetime, time, date, timezone, timedelta
from decimal import Context
from .const import (
def read_timestamp_millis(data, writer_schema=None, reader_schema=None):
    return epoch + timedelta(microseconds=data * 1000)