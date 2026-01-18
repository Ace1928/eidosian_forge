from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def operate_vrf_af(self):
    """config/delete vrf"""
    vrf_target_operate = ''
    if self.route_distinguisher is None:
        route_d = ''
    else:
        route_d = self.route_distinguisher
    if self.state == 'present':
        if self.vrf_aftype:
            if self.is_vrf_af_exist():
                self.vrf_af_type_changed = False
            else:
                self.vrf_af_type_changed = True
                configxmlstr = CE_NC_CREATE_VRF_AF % (self.vrf, self.vrf_aftype, route_d, vrf_target_operate)
        else:
            self.vrf_af_type_changed = bool(self.is_vrf_af_exist())
        if self.vpn_target_state == 'present':
            if self.evpn is False and (not self.is_vrf_rt_exist()):
                vrf_target_operate = CE_NC_CREATE_VRF_TARGET % (self.vpn_target_type, self.vpn_target_value)
                configxmlstr = CE_NC_CREATE_VRF_AF % (self.vrf, self.vrf_aftype, route_d, vrf_target_operate)
                self.vpn_target_changed = True
            if self.evpn is True and (not self.is_vrf_rt_exist()):
                vrf_target_operate = CE_NC_CREATE_EXTEND_VRF_TARGET % (self.vpn_target_type, self.vpn_target_value)
                configxmlstr = CE_NC_CREATE_VRF_AF % (self.vrf, self.vrf_aftype, route_d, vrf_target_operate)
                self.vpn_target_changed = True
        elif self.vpn_target_state == 'absent':
            if self.evpn is False and self.is_vrf_rt_exist():
                vrf_target_operate = CE_NC_DELETE_VRF_TARGET % (self.vpn_target_type, self.vpn_target_value)
                configxmlstr = CE_NC_CREATE_VRF_AF % (self.vrf, self.vrf_aftype, route_d, vrf_target_operate)
                self.vpn_target_changed = True
            if self.evpn is True and self.is_vrf_rt_exist():
                vrf_target_operate = CE_NC_DELETE_EXTEND_VRF_TARGET % (self.vpn_target_type, self.vpn_target_value)
                configxmlstr = CE_NC_CREATE_VRF_AF % (self.vrf, self.vrf_aftype, route_d, vrf_target_operate)
                self.vpn_target_changed = True
        elif self.route_distinguisher:
            if not self.is_vrf_rd_exist():
                configxmlstr = CE_NC_CREATE_VRF_AF % (self.vrf, self.vrf_aftype, route_d, vrf_target_operate)
                self.vrf_rd_changed = True
            else:
                self.vrf_rd_changed = False
        elif self.is_vrf_rd_exist():
            configxmlstr = CE_NC_CREATE_VRF_AF % (self.vrf, self.vrf_aftype, route_d, vrf_target_operate)
            self.vrf_rd_changed = True
        else:
            self.vrf_rd_changed = False
        if not self.vrf_rd_changed and (not self.vrf_af_type_changed) and (not self.vpn_target_changed):
            self.changed = False
        else:
            self.changed = True
    elif self.is_vrf_af_exist():
        configxmlstr = CE_NC_DELETE_VRF_AF % (self.vrf, self.vrf_aftype)
        self.changed = True
    else:
        self.changed = False
    if not self.changed:
        return
    conf_str = build_config_xml(configxmlstr)
    recv_xml = set_nc_config(self.module, conf_str)
    self.check_response(recv_xml, 'OPERATE_VRF_AF')