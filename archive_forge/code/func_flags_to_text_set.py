import base64
import struct
import dns.exception
import dns.dnssec
import dns.rdata
def flags_to_text_set(self):
    """Convert a DNSKEY flags value to set texts
        @rtype: set([string])"""
    return flags_to_text_set(self.flags)