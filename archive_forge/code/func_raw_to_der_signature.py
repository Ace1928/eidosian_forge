import base64
import binascii
import re
from typing import Union
def raw_to_der_signature(raw_sig: bytes, curve: 'EllipticCurve') -> bytes:
    num_bits = curve.key_size
    num_bytes = (num_bits + 7) // 8
    if len(raw_sig) != 2 * num_bytes:
        raise ValueError('Invalid signature')
    r = bytes_to_number(raw_sig[:num_bytes])
    s = bytes_to_number(raw_sig[num_bytes:])
    return bytes(encode_dss_signature(r, s))