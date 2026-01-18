import re
from .checks import check_data, check_msgdict, check_value
from .decode import decode_message
from .encode import encode_message
from .specs import REALTIME_TYPES, SPEC_BY_TYPE, make_msgdict
from .strings import msg2str, str2msg
def format_as_string(msg, include_time=True):
    """Format a message and return as a string.

    This is equivalent to str(message).

    To leave out the time attribute, pass include_time=False.
    """
    return msg2str(vars(msg), include_time=include_time)