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
def test_ipv4_udp(self):
    e = ethernet.ethernet(self.dst_mac, self.src_mac, ether.ETH_TYPE_IP)
    ip = ipv4.ipv4(4, 5, 1, 0, 3, 1, 4, 64, inet.IPPROTO_UDP, 0, self.src_ip, self.dst_ip)
    u = udp.udp(6415, 8080, 0, 0)
    p = packet.Packet()
    p.add_protocol(e)
    p.add_protocol(ip)
    p.add_protocol(u)
    p.add_protocol(self.payload)
    p.serialize()
    e_buf = self.dst_mac_bin + self.src_mac_bin + b'\x08\x00'
    ip_buf = b'E' + b'\x01' + b'\x00<' + b'\x00\x03' + b' \x04' + b'@' + b'\x11' + b'\x00\x00' + self.src_ip_bin + self.dst_ip_bin
    u_buf = b'\x19\x0f' + b'\x1f\x90' + b'\x00(' + b'\x00\x00'
    buf = e_buf + ip_buf + u_buf + self.payload
    pkt = packet.Packet(p.data)
    protocols = self.get_protocols(pkt)
    p_eth = protocols['ethernet']
    p_ipv4 = protocols['ipv4']
    p_udp = protocols['udp']
    self.assertTrue(p_eth)
    self.assertEqual(self.dst_mac, p_eth.dst)
    self.assertEqual(self.src_mac, p_eth.src)
    self.assertEqual(ether.ETH_TYPE_IP, p_eth.ethertype)
    self.assertTrue(p_ipv4)
    self.assertEqual(4, p_ipv4.version)
    self.assertEqual(5, p_ipv4.header_length)
    self.assertEqual(1, p_ipv4.tos)
    l = len(ip_buf) + len(u_buf) + len(self.payload)
    self.assertEqual(l, p_ipv4.total_length)
    self.assertEqual(3, p_ipv4.identification)
    self.assertEqual(1, p_ipv4.flags)
    self.assertEqual(64, p_ipv4.ttl)
    self.assertEqual(inet.IPPROTO_UDP, p_ipv4.proto)
    self.assertEqual(self.src_ip, p_ipv4.src)
    self.assertEqual(self.dst_ip, p_ipv4.dst)
    t = bytearray(ip_buf)
    struct.pack_into('!H', t, 10, p_ipv4.csum)
    self.assertEqual(packet_utils.checksum(t), 0)
    self.assertTrue(p_udp)
    self.assertEqual(6415, p_udp.src_port)
    self.assertEqual(8080, p_udp.dst_port)
    self.assertEqual(len(u_buf) + len(self.payload), p_udp.total_length)
    self.assertEqual(30642, p_udp.csum)
    t = bytearray(u_buf)
    struct.pack_into('!H', t, 6, p_udp.csum)
    ph = struct.pack('!4s4sBBH', self.src_ip_bin, self.dst_ip_bin, 0, 17, len(u_buf) + len(self.payload))
    t = ph + t + self.payload
    self.assertEqual(packet_utils.checksum(t), 0)
    self.assertTrue('payload' in protocols)
    self.assertEqual(self.payload, protocols['payload'])
    eth_values = {'dst': self.dst_mac, 'src': self.src_mac, 'ethertype': ether.ETH_TYPE_IP}
    _eth_str = ','.join(['%s=%s' % (k, repr(eth_values[k])) for k, v in inspect.getmembers(p_eth) if k in eth_values])
    eth_str = '%s(%s)' % (ethernet.ethernet.__name__, _eth_str)
    ipv4_values = {'version': 4, 'header_length': 5, 'tos': 1, 'total_length': l, 'identification': 3, 'flags': 1, 'offset': p_ipv4.offset, 'ttl': 64, 'proto': inet.IPPROTO_UDP, 'csum': p_ipv4.csum, 'src': self.src_ip, 'dst': self.dst_ip, 'option': None}
    _ipv4_str = ','.join(['%s=%s' % (k, repr(ipv4_values[k])) for k, v in inspect.getmembers(p_ipv4) if k in ipv4_values])
    ipv4_str = '%s(%s)' % (ipv4.ipv4.__name__, _ipv4_str)
    udp_values = {'src_port': 6415, 'dst_port': 8080, 'total_length': len(u_buf) + len(self.payload), 'csum': 30642}
    _udp_str = ','.join(['%s=%s' % (k, repr(udp_values[k])) for k, v in inspect.getmembers(p_udp) if k in udp_values])
    udp_str = '%s(%s)' % (udp.udp.__name__, _udp_str)
    pkt_str = '%s, %s, %s, %s' % (eth_str, ipv4_str, udp_str, repr(protocols['payload']))
    self.assertEqual(eth_str, str(p_eth))
    self.assertEqual(eth_str, repr(p_eth))
    self.assertEqual(ipv4_str, str(p_ipv4))
    self.assertEqual(ipv4_str, repr(p_ipv4))
    self.assertEqual(udp_str, str(p_udp))
    self.assertEqual(udp_str, repr(p_udp))
    self.assertEqual(pkt_str, str(pkt))
    self.assertEqual(pkt_str, repr(pkt))