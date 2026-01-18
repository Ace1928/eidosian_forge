import base64
import calendar
import hashlib
import hmac
import os
import struct
import time
import ntlm_auth.compute_hash as comphash
import ntlm_auth.compute_keys as compkeys
import ntlm_auth.messages
from ntlm_auth.des import DES
from ntlm_auth.constants import AvId, AvFlags, NegotiateFlags
from ntlm_auth.gss_channel_bindings import GssChannelBindingsStruct
def get_windows_timestamp():
    seconds_since_origin = 116444736000 + calendar.timegm(time.gmtime())
    timestamp = struct.pack('<q', seconds_since_origin * 10000000)
    return timestamp