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
def test_ipv6_tcp(self):
    e = ethernet.ethernet(ethertype=ether.ETH_TYPE_IPV6)
    ip = ipv6.ipv6()
    t = tcp.tcp(option=b'\x01\x02')
    p = e / ip / t / self.payload
    p.serialize()
    ipaddr = addrconv.ipv6.text_to_bin('::')
    e_buf = b'\xff\xff\xff\xff\xff\xff' + b'\x00\x00\x00\x00\x00\x00' + b'\x86\xdd'
    ip_buf = b'`\x00\x00\x00' + b'\x00\x00' + b'\x06' + b'\xff' + b'\x00\x00' + ipaddr + ipaddr
    t_buf = b'\x00\x00' + b'\x00\x00' + b'\x00\x00\x00\x00' + b'\x00\x00\x00\x00' + b'`' + b'\x00' + b'\x00\x00' + b'\x00\x00' + b'\x00\x00' + b'\x01\x02\x00\x00'
    buf = e_buf + ip_buf + t_buf + self.payload
    pkt = packet.Packet(p.data)
    protocols = self.get_protocols(pkt)
    p_eth = protocols['ethernet']
    p_ipv6 = protocols['ipv6']
    p_tcp = protocols['tcp']
    self.assertTrue(p_eth)
    self.assertEqual('ff:ff:ff:ff:ff:ff', p_eth.dst)
    self.assertEqual('00:00:00:00:00:00', p_eth.src)
    self.assertEqual(ether.ETH_TYPE_IPV6, p_eth.ethertype)
    self.assertTrue(p_ipv6)
    self.assertEqual(6, p_ipv6.version)
    self.assertEqual(0, p_ipv6.traffic_class)
    self.assertEqual(0, p_ipv6.flow_label)
    self.assertEqual(len(t_buf) + len(self.payload), p_ipv6.payload_length)
    self.assertEqual(inet.IPPROTO_TCP, p_ipv6.nxt)
    self.assertEqual(255, p_ipv6.hop_limit)
    self.assertEqual('10::10', p_ipv6.src)
    self.assertEqual('20::20', p_ipv6.dst)
    self.assertTrue(p_tcp)
    self.assertEqual(1, p_tcp.src_port)
    self.assertEqual(1, p_tcp.dst_port)
    self.assertEqual(0, p_tcp.seq)
    self.assertEqual(0, p_tcp.ack)
    self.assertEqual(6, p_tcp.offset)
    self.assertEqual(0, p_tcp.bits)
    self.assertEqual(0, p_tcp.window_size)
    self.assertEqual(0, p_tcp.urgent)
    self.assertEqual(len(t_buf), len(p_tcp))
    t = bytearray(t_buf)
    struct.pack_into('!H', t, 16, p_tcp.csum)
    ph = struct.pack('!16s16sI3xB', ipaddr, ipaddr, len(t_buf) + len(self.payload), 6)
    t = ph + t + self.payload
    self.assertEqual(packet_utils.checksum(t), 98)
    self.assertTrue('payload' in protocols)
    self.assertEqual(self.payload, protocols['payload'])
    eth_values = {'dst': 'ff:ff:ff:ff:ff:ff', 'src': '00:00:00:00:00:00', 'ethertype': ether.ETH_TYPE_IPV6}
    _eth_str = ','.join(['%s=%s' % (k, repr(eth_values[k])) for k, v in inspect.getmembers(p_eth) if k in eth_values])
    eth_str = '%s(%s)' % (ethernet.ethernet.__name__, _eth_str)
    ipv6_values = {'version': 6, 'traffic_class': 0, 'flow_label': 0, 'payload_length': len(t_buf) + len(self.payload), 'nxt': inet.IPPROTO_TCP, 'hop_limit': 255, 'src': '10::10', 'dst': '20::20', 'ext_hdrs': []}
    _ipv6_str = ','.join(['%s=%s' % (k, repr(ipv6_values[k])) for k, v in inspect.getmembers(p_ipv6) if k in ipv6_values])
    ipv6_str = '%s(%s)' % (ipv6.ipv6.__name__, _ipv6_str)
    tcp_values = {'src_port': 1, 'dst_port': 1, 'seq': 0, 'ack': 0, 'offset': 6, 'bits': 0, 'window_size': 0, 'csum': p_tcp.csum, 'urgent': 0, 'option': p_tcp.option}
    _tcp_str = ','.join(['%s=%s' % (k, repr(tcp_values[k])) for k, v in inspect.getmembers(p_tcp) if k in tcp_values])
    tcp_str = '%s(%s)' % (tcp.tcp.__name__, _tcp_str)
    pkt_str = '%s, %s, %s, %s' % (eth_str, ipv6_str, tcp_str, repr(protocols['payload']))
    self.assertEqual(eth_str, str(p_eth))
    self.assertEqual(eth_str, repr(p_eth))
    self.assertEqual(ipv6_str, str(p_ipv6))
    self.assertEqual(ipv6_str, repr(p_ipv6))
    self.assertEqual(tcp_str, str(p_tcp))
    self.assertEqual(tcp_str, repr(p_tcp))
    self.assertEqual(pkt_str, str(pkt))
    self.assertEqual(pkt_str, repr(pkt))