import datetime
import logging
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.exception import OFPUnknownVersion
from os_ken.lib import hub
from os_ken.lib import mac
from os_ken.lib.dpid import dpid_to_str
from os_ken.lib.packet import bpdu
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import llc
from os_ken.lib.packet import packet
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
def rcv_config_bpdu(self, bpdu_pkt):
    root_id = BridgeId(bpdu_pkt.root_priority, bpdu_pkt.root_system_id_extension, bpdu_pkt.root_mac_address)
    root_path_cost = bpdu_pkt.root_path_cost
    designated_bridge_id = BridgeId(bpdu_pkt.bridge_priority, bpdu_pkt.bridge_system_id_extension, bpdu_pkt.bridge_mac_address)
    designated_port_id = PortId(bpdu_pkt.port_priority, bpdu_pkt.port_number)
    msg_priority = Priority(root_id, root_path_cost, designated_bridge_id, designated_port_id)
    msg_times = Times(bpdu_pkt.message_age, bpdu_pkt.max_age, bpdu_pkt.hello_time, bpdu_pkt.forward_delay)
    rcv_info = Stp.compare_bpdu_info(self.designated_priority, self.designated_times, msg_priority, msg_times)
    if rcv_info is SUPERIOR:
        self.designated_priority = msg_priority
        self.designated_times = msg_times
    chk_flg = False
    if (rcv_info is SUPERIOR or rcv_info is REPEATED) and (self.role is ROOT_PORT or self.role is NON_DESIGNATED_PORT):
        self._update_wait_bpdu_timer()
        chk_flg = True
    elif rcv_info is INFERIOR and self.role is DESIGNATED_PORT:
        chk_flg = True
    rcv_tc = False
    if chk_flg:
        tc_flag_mask = 1
        tcack_flag_mask = 128
        if bpdu_pkt.flags & tc_flag_mask:
            self.logger.debug('[port=%d] receive TopologyChange BPDU.', self.ofport.port_no, extra=self.dpid_str)
            rcv_tc = True
        if bpdu_pkt.flags & tcack_flag_mask:
            self.logger.debug('[port=%d] receive TopologyChangeAck BPDU.', self.ofport.port_no, extra=self.dpid_str)
            if self.send_tcn_flg:
                self.send_tcn_flg = False
    return (rcv_info, rcv_tc)