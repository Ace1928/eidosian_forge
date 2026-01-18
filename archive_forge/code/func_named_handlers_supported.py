import logging.handlers
import re
import sys
import types
def named_handlers_supported():
    major, minor = sys.version_info[:2]
    if major == 2:
        result = minor >= 7
    elif major == 3:
        result = minor >= 2
    else:
        result = major > 3
    return result