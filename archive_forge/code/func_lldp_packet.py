import logging
import struct
import time
from os_ken import cfg
from collections import defaultdict
from os_ken.topology import event
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from os_ken.exception import OSKenException
from os_ken.lib import addrconv, hub
from os_ken.lib.mac import DONTCARE_STR
from os_ken.lib.dpid import dpid_to_str, str_to_dpid
from os_ken.lib.port_no import port_no_to_str
from os_ken.lib.packet import packet, ethernet
from os_ken.lib.packet import lldp, ether_types
from os_ken.ofproto.ether import ETH_TYPE_LLDP
from os_ken.ofproto.ether import ETH_TYPE_CFM
from os_ken.ofproto import nx_match
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_4
@staticmethod
def lldp_packet(dpid, port_no, dl_addr, ttl):
    pkt = packet.Packet()
    dst = lldp.LLDP_MAC_NEAREST_BRIDGE
    src = dl_addr
    ethertype = ETH_TYPE_LLDP
    eth_pkt = ethernet.ethernet(dst, src, ethertype)
    pkt.add_protocol(eth_pkt)
    tlv_chassis_id = lldp.ChassisID(subtype=lldp.ChassisID.SUB_LOCALLY_ASSIGNED, chassis_id=(LLDPPacket.CHASSIS_ID_FMT % dpid_to_str(dpid)).encode('ascii'))
    tlv_port_id = lldp.PortID(subtype=lldp.PortID.SUB_PORT_COMPONENT, port_id=struct.pack(LLDPPacket.PORT_ID_STR, port_no))
    tlv_ttl = lldp.TTL(ttl=ttl)
    tlv_end = lldp.End()
    tlvs = (tlv_chassis_id, tlv_port_id, tlv_ttl, tlv_end)
    lldp_pkt = lldp.lldp(tlvs)
    pkt.add_protocol(lldp_pkt)
    pkt.serialize()
    return pkt.data