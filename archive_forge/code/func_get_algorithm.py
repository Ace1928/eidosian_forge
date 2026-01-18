import hashlib
import hmac
import struct
import dns.exception
import dns.rdataclass
import dns.name
from ._compat import long, string_types, text_type
def get_algorithm(algorithm):
    """Returns the wire format string and the hash module to use for the
    specified TSIG algorithm

    @rtype: (string, hash constructor)
    @raises NotImplementedError: I{algorithm} is not supported
    """
    if isinstance(algorithm, string_types):
        algorithm = dns.name.from_text(algorithm)
    try:
        return (algorithm.to_digestable(), _hashes[algorithm])
    except KeyError:
        raise NotImplementedError('TSIG algorithm ' + str(algorithm) + ' is not supported')