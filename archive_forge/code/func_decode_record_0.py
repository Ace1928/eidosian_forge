import time
import logging
import datetime
import threading
from pyzor.engines.common import Record, DBHandle, BaseEngine
@staticmethod
def decode_record_0(s):
    r = Record()
    parts = s.split(',')
    fields = ('r_count', 'r_entered', 'r_updated')
    assert len(parts) == len(fields)
    for i in range(len(parts)):
        setattr(r, fields[i], int(parts[i]))
    return r