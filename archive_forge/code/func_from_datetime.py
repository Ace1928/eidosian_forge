import datetime
import decimal
from .base import Database
@classmethod
def from_datetime(cls, dt):
    return Oracle_datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)