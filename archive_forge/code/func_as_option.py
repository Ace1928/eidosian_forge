import struct
import dns.edns
import dns.exception
import dns.immutable
import dns.rdata
def as_option(option):
    if not isinstance(option, dns.edns.Option):
        raise ValueError('option is not a dns.edns.option')
    return option