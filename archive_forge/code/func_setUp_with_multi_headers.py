import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
def setUp_with_multi_headers(self):
    self.opt1_type = 5
    self.opt1_len = 2
    self.opt1_data = b'\x00\x00'
    self.opt2_type = 1
    self.opt2_len = 0
    self.opt2_data = None
    self.options = [ipv6.option(self.opt1_type, self.opt1_len, self.opt1_data), ipv6.option(self.opt2_type, self.opt2_len, self.opt2_data)]
    self.hop_opts_nxt = ipv6.auth.TYPE
    self.hop_opts_size = 0
    self.hop_opts = ipv6.hop_opts(self.hop_opts_nxt, self.hop_opts_size, self.options)
    self.auth_nxt = 6
    self.auth_size = 4
    self.auth_spi = 256
    self.auth_seq = 1
    self.auth_data = b'\xa0\xe7\xf8\xab\xf9i\x1a\x8b\xf3\x9f|\xae'
    self.auth = ipv6.auth(self.auth_nxt, self.auth_size, self.auth_spi, self.auth_seq, self.auth_data)
    self.ext_hdrs = [self.hop_opts, self.auth]
    self.payload_length += len(self.hop_opts) + len(self.auth)
    self.nxt = ipv6.hop_opts.TYPE
    self.ip = ipv6.ipv6(self.version, self.traffic_class, self.flow_label, self.payload_length, self.nxt, self.hop_limit, self.src, self.dst, self.ext_hdrs)
    self.buf = struct.pack(ipv6.ipv6._PACK_STR, self.v_tc_flow, self.payload_length, self.nxt, self.hop_limit, addrconv.ipv6.text_to_bin(self.src), addrconv.ipv6.text_to_bin(self.dst))
    self.buf += self.hop_opts.serialize()
    self.buf += self.auth.serialize()