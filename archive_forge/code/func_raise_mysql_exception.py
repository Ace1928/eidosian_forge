import struct
from .constants import ER
def raise_mysql_exception(data):
    errno = struct.unpack('<h', data[1:3])[0]
    errval = data[9:].decode('utf-8', 'replace')
    errorclass = error_map.get(errno)
    if errorclass is None:
        errorclass = InternalError if errno < 1000 else OperationalError
    raise errorclass(errno, errval)