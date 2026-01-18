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
def get_hits_for_key(self, key: CacheKey) -> int:
    """Return the number of cache hits associated with the specified key."""
    with self.lock:
        node = self.data.get(key)
        if node is None or node.value.expiration <= time.time():
            return 0
        else:
            return node.hits