from functools import reduce
import unittest
import testscenarios
from os_ken.ofproto import ofproto_v1_5
from os_ken.ofproto import ofproto_v1_5_parser
def flatten_one(l, i):
    if isinstance(i, tuple):
        return l + flatten(i)
    else:
        return l + [i]