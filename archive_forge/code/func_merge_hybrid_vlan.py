from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def merge_hybrid_vlan(self, ifname, pvid_vlan, tagged_vlans, untagged_vlans):
    """Merge hybrid interface vlan"""
    change = False
    xmlstr = ''
    pvid = ''
    tagged = ''
    untagged = ''
    self.updates_cmd.append('interface %s' % ifname)
    if tagged_vlans:
        vlan_targed_list = self.vlan_range_to_list(tagged_vlans)
        vlan_targed_map = self.vlan_list_to_bitmap(vlan_targed_list)
    if untagged_vlans:
        vlan_untarged_list = self.vlan_range_to_list(untagged_vlans)
        vlan_untarged_map = self.vlan_list_to_bitmap(vlan_untarged_list)
    if self.state == 'present':
        if self.intf_info['linkType'] == 'hybrid':
            if pvid_vlan and self.intf_info['pvid'] != pvid_vlan:
                self.updates_cmd.append('port hybrid pvid vlan %s' % pvid_vlan)
                pvid = pvid_vlan
                change = True
            if tagged_vlans:
                add_vlans = self.vlan_bitmap_add(self.intf_info['trunkVlans'], vlan_targed_map)
                if not is_vlan_bitmap_empty(add_vlans):
                    self.updates_cmd.append('port hybrid tagged vlan %s' % tagged_vlans.replace(',', ' ').replace('-', ' to '))
                    tagged = '%s:%s' % (add_vlans, add_vlans)
                    change = True
            if untagged_vlans:
                add_vlans = self.vlan_bitmap_add(self.intf_info['untagVlans'], vlan_untarged_map)
                if not is_vlan_bitmap_empty(add_vlans):
                    self.updates_cmd.append('port hybrid untagged vlan %s' % untagged_vlans.replace(',', ' ').replace('-', ' to '))
                    untagged = '%s:%s' % (add_vlans, add_vlans)
                    change = True
            if pvid or tagged or untagged:
                xmlstr += CE_NC_SET_PORT % (ifname, 'hybrid', pvid, tagged, untagged)
                if not pvid:
                    xmlstr = xmlstr.replace('<pvid></pvid>', '')
                if not tagged:
                    xmlstr = xmlstr.replace('<trunkVlans></trunkVlans>', '')
                if not untagged:
                    xmlstr = xmlstr.replace('<untagVlans></untagVlans>', '')
        else:
            self.updates_cmd.append('port link-type hybrid')
            change = True
            if pvid_vlan:
                self.updates_cmd.append('port hybrid pvid vlan %s' % pvid_vlan)
                pvid = pvid_vlan
            if tagged_vlans:
                self.updates_cmd.append('port hybrid tagged vlan %s' % tagged_vlans.replace(',', ' ').replace('-', ' to '))
                tagged = '%s:%s' % (vlan_targed_map, vlan_targed_map)
            if untagged_vlans:
                self.updates_cmd.append('port hybrid untagged vlan %s' % untagged_vlans.replace(',', ' ').replace('-', ' to '))
                untagged = '%s:%s' % (vlan_untarged_map, vlan_untarged_map)
            if pvid or tagged or untagged:
                xmlstr += CE_NC_SET_PORT % (ifname, 'hybrid', pvid, tagged, untagged)
                if not pvid:
                    xmlstr = xmlstr.replace('<pvid></pvid>', '')
                if not tagged:
                    xmlstr = xmlstr.replace('<trunkVlans></trunkVlans>', '')
                if not untagged:
                    xmlstr = xmlstr.replace('<untagVlans></untagVlans>', '')
            if not pvid_vlan and (not tagged_vlans) and (not untagged_vlans):
                xmlstr += CE_NC_SET_PORT_MODE % (ifname, 'hybrid')
                self.updates_cmd.append('undo port hybrid untagged vlan 1')
    elif self.state == 'absent':
        if self.intf_info['linkType'] == 'hybrid':
            if pvid_vlan and self.intf_info['pvid'] == pvid_vlan and (pvid_vlan != '1'):
                self.updates_cmd.append('undo port hybrid pvid vlan %s' % pvid_vlan)
                pvid = '1'
                change = True
            if tagged_vlans:
                del_vlans = self.vlan_bitmap_del(self.intf_info['trunkVlans'], vlan_targed_map)
                if not is_vlan_bitmap_empty(del_vlans):
                    self.updates_cmd.append('undo port hybrid tagged vlan %s' % tagged_vlans.replace(',', ' ').replace('-', ' to '))
                    undo_map = vlan_bitmap_undo(del_vlans)
                    tagged = '%s:%s' % (undo_map, del_vlans)
                    change = True
            if untagged_vlans:
                del_vlans = self.vlan_bitmap_del(self.intf_info['untagVlans'], vlan_untarged_map)
                if not is_vlan_bitmap_empty(del_vlans):
                    self.updates_cmd.append('undo port hybrid untagged vlan %s' % untagged_vlans.replace(',', ' ').replace('-', ' to '))
                    undo_map = vlan_bitmap_undo(del_vlans)
                    untagged = '%s:%s' % (undo_map, del_vlans)
                    change = True
            if pvid or tagged or untagged:
                xmlstr += CE_NC_SET_PORT % (ifname, 'hybrid', pvid, tagged, untagged)
                if not pvid:
                    xmlstr = xmlstr.replace('<pvid></pvid>', '')
                if not tagged:
                    xmlstr = xmlstr.replace('<trunkVlans></trunkVlans>', '')
                if not untagged:
                    xmlstr = xmlstr.replace('<untagVlans></untagVlans>', '')
    if not change:
        self.updates_cmd.pop()
        return
    conf_str = '<config>' + xmlstr + '</config>'
    rcv_xml = set_nc_config(self.module, conf_str)
    self.check_response(rcv_xml, 'MERGE_HYBRID_PORT')
    self.changed = True