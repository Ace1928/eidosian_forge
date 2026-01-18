import re
from .checks import check_data, check_msgdict, check_value
from .decode import decode_message
from .encode import encode_message
from .specs import REALTIME_TYPES, SPEC_BY_TYPE, make_msgdict
from .strings import msg2str, str2msg
@classmethod
def from_hex(cl, text, time=0, sep=None):
    """Parse a hex encoded message.

        This is the reverse of msg.hex().
        """
    text = re.sub('\\s', ' ', text)
    if sep is not None:
        text = text.replace(sep, ' ' * len(sep))
    return cl.from_bytes(bytearray.fromhex(text), time=time)