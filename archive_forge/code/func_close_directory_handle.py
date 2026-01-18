from __future__ import with_statement
import ctypes.wintypes
from functools import reduce
def close_directory_handle(handle):
    try:
        CancelIoEx(handle, None)
        CloseHandle(handle)
    except WindowsError:
        try:
            CloseHandle(handle)
        except:
            return