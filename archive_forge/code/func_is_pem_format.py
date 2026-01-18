import base64
import binascii
import re
from typing import Union
def is_pem_format(key: bytes) -> bool:
    return bool(_PEM_RE.search(key))