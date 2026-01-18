import collections
import random
import struct
from typing import Any, List
import dns.exception
import dns.ipv4
import dns.ipv6
import dns.name
import dns.rdata
def parse_formatted_hex(formatted, num_chunks, chunk_size, separator):
    if len(formatted) != num_chunks * (chunk_size + 1) - 1:
        raise ValueError('invalid formatted hex string')
    value = b''
    for _ in range(num_chunks):
        chunk = formatted[0:chunk_size]
        value += int(chunk, 16).to_bytes(chunk_size // 2, 'big')
        formatted = formatted[chunk_size:]
        if len(formatted) > 0 and formatted[0] != separator:
            raise ValueError('invalid formatted hex string')
        formatted = formatted[1:]
    return value