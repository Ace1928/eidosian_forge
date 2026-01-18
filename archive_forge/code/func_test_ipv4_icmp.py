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
def test_ipv4_icmp(self):
    e = ethernet.ethernet()
    ip = ipv4.ipv4(proto=inet.IPPROTO_ICMP)
    ic = icmp.icmp()
    p = e / ip / ic
    p.serialize()
    ipaddr = addrconv.ipv4.text_to_bin('0.0.0.0')
    e_buf = b'\xff\xff\xff\xff\xff\xff' + b'\x00\x00\x00\x00\x00\x00' + b'\x08\x00'
    ip_buf = b'E' + b'\x00' + b'\x00\x1c' + b'\x00\x00' + b'\x00\x00' + b'\xff' + b'\x01' + b'\x00\x00' + ipaddr + ipaddr
    ic_buf = b'\x08' + b'\x00' + b'\x00\x00' + b'\x00\x00' + b'\x00\x00'
    buf = e_buf + ip_buf + ic_buf
    pkt = packet.Packet(p.data)
    protocols = self.get_protocols(pkt)
    p_eth = protocols['ethernet']
    p_ipv4 = protocols['ipv4']
    p_icmp = protocols['icmp']
    self.assertTrue(p_eth)
    self.assertEqual('ff:ff:ff:ff:ff:ff', p_eth.dst)
    self.assertEqual('00:00:00:00:00:00', p_eth.src)
    self.assertEqual(ether.ETH_TYPE_IP, p_eth.ethertype)
    self.assertTrue(p_ipv4)
    self.assertEqual(4, p_ipv4.version)
    self.assertEqual(5, p_ipv4.header_length)
    self.assertEqual(0, p_ipv4.tos)
    l = len(ip_buf) + len(ic_buf)
    self.assertEqual(l, p_ipv4.total_length)
    self.assertEqual(0, p_ipv4.identification)
    self.assertEqual(0, p_ipv4.flags)
    self.assertEqual(255, p_ipv4.ttl)
    self.assertEqual(inet.IPPROTO_ICMP, p_ipv4.proto)
    self.assertEqual('10.0.0.1', p_ipv4.src)
    self.assertEqual('10.0.0.2', p_ipv4.dst)
    t = bytearray(ip_buf)
    struct.pack_into('!H', t, 10, p_ipv4.csum)
    self.assertEqual(packet_utils.checksum(t), 5123)
    self.assertTrue(p_icmp)
    self.assertEqual(8, p_icmp.type)
    self.assertEqual(0, p_icmp.code)
    self.assertEqual(0, p_icmp.data.id)
    self.assertEqual(0, p_icmp.data.seq)
    self.assertEqual(len(ic_buf), len(p_icmp))
    t = bytearray(ic_buf)
    struct.pack_into('!H', t, 2, p_icmp.csum)
    self.assertEqual(packet_utils.checksum(t), 0)
    eth_values = {'dst': 'ff:ff:ff:ff:ff:ff', 'src': '00:00:00:00:00:00', 'ethertype': ether.ETH_TYPE_IP}
    _eth_str = ','.join(['%s=%s' % (k, repr(eth_values[k])) for k, _ in inspect.getmembers(p_eth) if k in eth_values])
    eth_str = '%s(%s)' % (ethernet.ethernet.__name__, _eth_str)
    ipv4_values = {'version': 4, 'header_length': 5, 'tos': 0, 'total_length': l, 'identification': 0, 'flags': 0, 'offset': p_ipv4.offset, 'ttl': 255, 'proto': inet.IPPROTO_ICMP, 'csum': p_ipv4.csum, 'src': '10.0.0.1', 'dst': '10.0.0.2', 'option': None}
    _ipv4_str = ','.join(['%s=%s' % (k, repr(ipv4_values[k])) for k, _ in inspect.getmembers(p_ipv4) if k in ipv4_values])
    ipv4_str = '%s(%s)' % (ipv4.ipv4.__name__, _ipv4_str)
    echo_values = {'id': 0, 'seq': 0, 'data': None}
    _echo_str = ','.join(['%s=%s' % (k, repr(echo_values[k])) for k in sorted(echo_values.keys())])
    echo_str = '%s(%s)' % (icmp.echo.__name__, _echo_str)
    icmp_values = {'type': 8, 'code': 0, 'csum': p_icmp.csum, 'data': echo_str}
    _icmp_str = ','.join(['%s=%s' % (k, icmp_values[k]) for k, _ in inspect.getmembers(p_icmp) if k in icmp_values])
    icmp_str = '%s(%s)' % (icmp.icmp.__name__, _icmp_str)
    pkt_str = '%s, %s, %s' % (eth_str, ipv4_str, icmp_str)
    self.assertEqual(eth_str, str(p_eth))
    self.assertEqual(eth_str, repr(p_eth))
    self.assertEqual(ipv4_str, str(p_ipv4))
    self.assertEqual(ipv4_str, repr(p_ipv4))
    self.assertEqual(icmp_str, str(p_icmp))
    self.assertEqual(icmp_str, repr(p_icmp))
    self.assertEqual(pkt_str, str(pkt))
    self.assertEqual(pkt_str, repr(pkt))