import ctypes
import ctypes.util
import sys
def get_host_error_message():
    """Return host error message."""
    buf = ctypes.create_string_buffer(PM_HOST_ERROR_MSG_LEN)
    lib.Pm_GetHostErrorText(buf, PM_HOST_ERROR_MSG_LEN)
    return buf.raw.decode().rstrip('\x00')