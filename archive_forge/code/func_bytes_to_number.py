import base64
import binascii
import re
from typing import Union
def bytes_to_number(string: bytes) -> int:
    return int(binascii.b2a_hex(string), 16)