import os
import re
def parse_addresses(s):
    for addr in s.split(';'):
        transport, info = addr.split(':', 1)
        kv = {}
        for x in info.split(','):
            k, v = x.split('=', 1)
            kv[k] = unescape(v)
        yield (transport, kv)