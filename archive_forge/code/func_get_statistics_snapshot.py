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
def get_statistics_snapshot(self) -> CacheStatistics:
    """Return a consistent snapshot of all the statistics.

        If running with multiple threads, it's better to take a
        snapshot than to call statistics methods such as hits() and
        misses() individually.
        """
    with self.lock:
        return self.statistics.clone()