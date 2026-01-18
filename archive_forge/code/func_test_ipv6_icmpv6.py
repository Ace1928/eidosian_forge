import unittest
import logging
import struct
import inspect
from os_ken.ofproto import ether, inet
from os_ken.lib.packet import arp
from os_ken.lib.packet import bpdu
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import icmp, icmpv6
from os_ken.lib.packet import ipv4, ipv6
from os_ken.lib.packet import llc
from os_ken.lib.packet import packet, packet_utils
from os_ken.lib.packet import sctp
from os_ken.lib.packet import tcp, udp
from os_ken.lib.packet import vlan
from os_ken.lib import addrconv
def test_ipv6_icmpv6(self):
    e = ethernet.ethernet(ethertype=ether.ETH_TYPE_IPV6)
    ip = ipv6.ipv6(nxt=inet.IPPROTO_ICMPV6)
    ic = icmpv6.icmpv6()
    p = e / ip / ic
    p.serialize()
    ipaddr = addrconv.ipv6.text_to_bin('::')
    e_buf = b'\xff\xff\xff\xff\xff\xff' + b'\x00\x00\x00\x00\x00\x00' + b'\x86\xdd'
    ip_buf = b'`\x00\x00\x00' + b'\x00\x00' + b':' + b'\xff' + b'\x00\x00' + ipaddr + ipaddr
    ic_buf = b'\x00' + b'\x00' + b'\x00\x00'
    buf = e_buf + ip_buf + ic_buf
    pkt = packet.Packet(p.data)
    protocols = self.get_protocols(pkt)
    p_eth = protocols['ethernet']
    p_ipv6 = protocols['ipv6']
    p_icmpv6 = protocols['icmpv6']
    self.assertTrue(p_eth)
    self.assertEqual('ff:ff:ff:ff:ff:ff', p_eth.dst)
    self.assertEqual('00:00:00:00:00:00', p_eth.src)
    self.assertEqual(ether.ETH_TYPE_IPV6, p_eth.ethertype)
    self.assertTrue(p_ipv6)
    self.assertEqual(6, p_ipv6.version)
    self.assertEqual(0, p_ipv6.traffic_class)
    self.assertEqual(0, p_ipv6.flow_label)
    self.assertEqual(len(ic_buf), p_ipv6.payload_length)
    self.assertEqual(inet.IPPROTO_ICMPV6, p_ipv6.nxt)
    self.assertEqual(255, p_ipv6.hop_limit)
    self.assertEqual('10::10', p_ipv6.src)
    self.assertEqual('20::20', p_ipv6.dst)
    self.assertTrue(p_icmpv6)
    self.assertEqual(0, p_icmpv6.type_)
    self.assertEqual(0, p_icmpv6.code)
    self.assertEqual(len(ic_buf), len(p_icmpv6))
    t = bytearray(ic_buf)
    struct.pack_into('!H', t, 2, p_icmpv6.csum)
    ph = struct.pack('!16s16sI3xB', ipaddr, ipaddr, len(ic_buf), 58)
    t = ph + t
    self.assertEqual(packet_utils.checksum(t), 96)
    eth_values = {'dst': 'ff:ff:ff:ff:ff:ff', 'src': '00:00:00:00:00:00', 'ethertype': ether.ETH_TYPE_IPV6}
    _eth_str = ','.join(['%s=%s' % (k, repr(eth_values[k])) for k, _ in inspect.getmembers(p_eth) if k in eth_values])
    eth_str = '%s(%s)' % (ethernet.ethernet.__name__, _eth_str)
    ipv6_values = {'version': 6, 'traffic_class': 0, 'flow_label': 0, 'payload_length': len(ic_buf), 'nxt': inet.IPPROTO_ICMPV6, 'hop_limit': 255, 'src': '10::10', 'dst': '20::20', 'ext_hdrs': []}
    _ipv6_str = ','.join(['%s=%s' % (k, repr(ipv6_values[k])) for k, _ in inspect.getmembers(p_ipv6) if k in ipv6_values])
    ipv6_str = '%s(%s)' % (ipv6.ipv6.__name__, _ipv6_str)
    icmpv6_values = {'type_': 0, 'code': 0, 'csum': p_icmpv6.csum, 'data': b''}
    _icmpv6_str = ','.join(['%s=%s' % (k, repr(icmpv6_values[k])) for k, _ in inspect.getmembers(p_icmpv6) if k in icmpv6_values])
    icmpv6_str = '%s(%s)' % (icmpv6.icmpv6.__name__, _icmpv6_str)
    pkt_str = '%s, %s, %s' % (eth_str, ipv6_str, icmpv6_str)
    self.assertEqual(eth_str, str(p_eth))
    self.assertEqual(eth_str, repr(p_eth))
    self.assertEqual(ipv6_str, str(p_ipv6))
    self.assertEqual(ipv6_str, repr(p_ipv6))
    self.assertEqual(icmpv6_str, str(p_icmpv6))
    self.assertEqual(icmpv6_str, repr(p_icmpv6))
    self.assertEqual(pkt_str, str(pkt))
    self.assertEqual(pkt_str, repr(pkt))