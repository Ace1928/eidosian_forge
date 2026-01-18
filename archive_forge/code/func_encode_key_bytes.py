import struct
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import dsa, utils
from dns.dnssecalgs.cryptography import CryptographyPrivateKey, CryptographyPublicKey
from dns.dnssectypes import Algorithm
from dns.rdtypes.ANY.DNSKEY import DNSKEY
def encode_key_bytes(self) -> bytes:
    """Encode a public key per RFC 2536, section 2."""
    pn = self.key.public_numbers()
    dsa_t = (self.key.key_size // 8 - 64) // 8
    if dsa_t > 8:
        raise ValueError('unsupported DSA key size')
    octets = 64 + dsa_t * 8
    res = struct.pack('!B', dsa_t)
    res += pn.parameter_numbers.q.to_bytes(20, 'big')
    res += pn.parameter_numbers.p.to_bytes(octets, 'big')
    res += pn.parameter_numbers.g.to_bytes(octets, 'big')
    res += pn.y.to_bytes(octets, 'big')
    return res