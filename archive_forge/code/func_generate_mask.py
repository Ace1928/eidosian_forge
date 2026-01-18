import functools
import itertools
import unittest
import testscenarios
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_5
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_3_parser
from os_ken.ofproto import ofproto_v1_4_parser
from os_ken.ofproto import ofproto_v1_5_parser
@classmethod
def generate_mask(cls):
    return list(cls.generate())[1]