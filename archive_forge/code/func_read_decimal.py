import uuid
from datetime import datetime, time, date, timezone, timedelta
from decimal import Context
from .const import (
def read_decimal(data, writer_schema=None, reader_schema=None):
    scale = writer_schema.get('scale', 0)
    precision = writer_schema['precision']
    unscaled_datum = int.from_bytes(data, byteorder='big', signed=True)
    decimal_context.prec = precision
    return decimal_context.create_decimal(unscaled_datum).scaleb(-scale, decimal_context)