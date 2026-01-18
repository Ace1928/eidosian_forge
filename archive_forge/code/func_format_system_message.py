import builtins
import ctypes.wintypes
from paramiko.util import u
def format_system_message(errno):
    """
    Call FormatMessage with a system error number to retrieve
    the descriptive error message.
    """
    ALLOCATE_BUFFER = 256
    FROM_SYSTEM = 4096
    flags = ALLOCATE_BUFFER | FROM_SYSTEM
    source = None
    message_id = errno
    language_id = 0
    result_buffer = ctypes.wintypes.LPWSTR()
    buffer_size = 0
    arguments = None
    bytes = ctypes.windll.kernel32.FormatMessageW(flags, source, message_id, language_id, ctypes.byref(result_buffer), buffer_size, arguments)
    handle_nonzero_success(bytes)
    message = result_buffer.value
    ctypes.windll.kernel32.LocalFree(result_buffer)
    return message