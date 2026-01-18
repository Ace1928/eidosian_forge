import base64
import binascii
import re
from typing import Union
def from_base64url_uint(val: Union[bytes, str]) -> int:
    data = base64url_decode(force_bytes(val))
    return int.from_bytes(data, byteorder='big')