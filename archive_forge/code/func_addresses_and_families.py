import contextlib
import random
import socket
import sys
import threading
import time
import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse
import dns._ddr
import dns.edns
import dns.exception
import dns.flags
import dns.inet
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.nameserver
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.rdtypes.svcbbase
import dns.reversename
import dns.tsig
def addresses_and_families(self, family: int=socket.AF_UNSPEC) -> Iterator[Tuple[str, int]]:
    if family == socket.AF_UNSPEC:
        yield from self.addresses_and_families(socket.AF_INET6)
        yield from self.addresses_and_families(socket.AF_INET)
        return
    elif family == socket.AF_INET6:
        answer = self.get(dns.rdatatype.AAAA)
    elif family == socket.AF_INET:
        answer = self.get(dns.rdatatype.A)
    else:
        raise NotImplementedError(f'unknown address family {family}')
    if answer:
        for rdata in answer:
            yield (rdata.address, family)