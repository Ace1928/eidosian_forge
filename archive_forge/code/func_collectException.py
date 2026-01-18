import sys
import traceback
import time
from io import StringIO
import linecache
from paste.exceptions import serial_number_generator
import warnings
def collectException(self, etype, value, tb, limit=None):
    __exception_formatter__ = 1
    frames = []
    ident_data = []
    traceback_decorators = []
    if limit is None:
        limit = self.getLimit()
    n = 0
    extra_data = {}
    while tb is not None and (limit is None or n < limit):
        if tb.tb_frame.f_locals.get('__exception_formatter__'):
            frames.append('(Recursive formatException() stopped)\n')
            break
        data = self.collectLine(tb, extra_data)
        frame = ExceptionFrame(**data)
        frames.append(frame)
        if frame.traceback_decorator is not None:
            traceback_decorators.append(frame.traceback_decorator)
        ident_data.append(frame.modname or '?')
        ident_data.append(frame.name or '?')
        tb = tb.tb_next
        n = n + 1
    ident_data.append(str(etype))
    ident = serial_number_generator.hash_identifier(' '.join(ident_data), length=5, upper=True, prefix=DEBUG_IDENT_PREFIX)
    result = CollectedException(frames=frames, exception_formatted=self.collectExceptionOnly(etype, value), exception_type=etype, exception_value=self.safeStr(value), identification_code=ident, date=time.localtime(), extra_data=extra_data)
    if etype is ImportError:
        extra_data['important', 'sys.path'] = [sys.path]
    for decorator in traceback_decorators:
        try:
            new_result = decorator(result)
            if new_result is not None:
                result = new_result
        except:
            pass
    return result