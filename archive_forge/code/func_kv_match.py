from __future__ import absolute_import, division, print_function
from collections import namedtuple
def kv_match(kvs, item):
    return all((item.get(kv.key) == kv.value for kv in kvs))