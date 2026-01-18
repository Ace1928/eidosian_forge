import collections
import random
import struct
from typing import Any, List
import dns.exception
import dns.ipv4
import dns.ipv6
import dns.name
import dns.rdata
def priority_processing_order(iterable):
    items = list(iterable)
    if len(items) == 1:
        return items
    by_priority = _priority_table(items)
    ordered = []
    for k in sorted(by_priority.keys()):
        rdatas = by_priority[k]
        random.shuffle(rdatas)
        ordered.extend(rdatas)
    return ordered