import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_error_reply(self, dp):
    ports = [0]
    for p in dp.ports:
        if p != dp.ofproto.OFPP_LOCAL:
            ports.append(p)
    port_no = max(ports) + 1
    self._verify = dp.ofproto.OFPT_ERROR
    m = dp.ofproto_parser.OFPPortMod(dp, port_no, 'ff:ff:ff:ff:ff:ff', 0, 0, 0)
    dp.send_msg(m)