import hashlib
import hmac
import struct
import dns.exception
import dns.rdataclass
import dns.name
from ._compat import long, string_types, text_type
def hmac_md5(wire, keyname, secret, time, fudge, original_id, error, other_data, request_mac, ctx=None, multi=False, first=True, algorithm=default_algorithm):
    return sign(wire, keyname, secret, time, fudge, original_id, error, other_data, request_mac, ctx, multi, first, algorithm)