import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
def setUp_cc_message(self):
    self.cc_message_md_lv = 1
    self.cc_message_version = 1
    self.cc_message_rdi = 1
    self.cc_message_interval = 1
    self.cc_message_seq_num = 123
    self.cc_message_mep_id = 4
    self.cc_message_md_name_format = 4
    self.cc_message_md_name_length = 0
    self.cc_message_md_name = b'hoge'
    self.cc_message_short_ma_name_format = 2
    self.cc_message_short_ma_name_length = 0
    self.cc_message_short_ma_name = b'pakeratta'
    self.cc_message_md_name_txfcf = 11
    self.cc_message_md_name_rxfcb = 22
    self.cc_message_md_name_txfcb = 33
    self.cc_message_tlvs = [cfm.sender_id_tlv(), cfm.port_status_tlv(), cfm.data_tlv(), cfm.interface_status_tlv(), cfm.reply_ingress_tlv(), cfm.reply_egress_tlv(), cfm.ltm_egress_identifier_tlv(), cfm.ltr_egress_identifier_tlv(), cfm.organization_specific_tlv()]
    self.message = cfm.cc_message(self.cc_message_md_lv, self.cc_message_version, self.cc_message_rdi, self.cc_message_interval, self.cc_message_seq_num, self.cc_message_mep_id, self.cc_message_md_name_format, self.cc_message_md_name_length, self.cc_message_md_name, self.cc_message_short_ma_name_format, self.cc_message_short_ma_name_length, self.cc_message_short_ma_name, self.cc_message_tlvs)
    self.ins = cfm.cfm(self.message)
    data = bytearray()
    prev = None
    self.buf = self.ins.serialize(data, prev)