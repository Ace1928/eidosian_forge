import re
from .checks import check_data, check_msgdict, check_value
from .decode import decode_message
from .encode import encode_message
from .specs import REALTIME_TYPES, SPEC_BY_TYPE, make_msgdict
from .strings import msg2str, str2msg
@property
def is_realtime(self):
    """True if the message is a system realtime message."""
    return self.type in REALTIME_TYPES