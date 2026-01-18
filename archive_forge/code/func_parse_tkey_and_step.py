import base64
import hashlib
import hmac
import struct
import dns.exception
import dns.name
import dns.rcode
import dns.rdataclass
@classmethod
def parse_tkey_and_step(cls, key, message, keyname):
    try:
        rrset = message.find_rrset(message.answer, keyname, dns.rdataclass.ANY, dns.rdatatype.TKEY)
        if rrset:
            token = rrset[0].key
            gssapi_context = key.secret
            return gssapi_context.step(token)
    except KeyError:
        pass