from abc import ABC, abstractmethod  # pylint: disable=no-name-in-module
from typing import Any, Optional, Type
import dns.rdataclass
import dns.rdatatype
from dns.dnssectypes import Algorithm
from dns.exception import AlgorithmKeyMismatch
from dns.rdtypes.ANY.DNSKEY import DNSKEY
from dns.rdtypes.dnskeybase import Flag
@classmethod
@abstractmethod
def from_pem(cls, private_pem: bytes, password: Optional[bytes]=None) -> 'GenericPrivateKey':
    """Create private key from PEM-encoded PKCS#8"""