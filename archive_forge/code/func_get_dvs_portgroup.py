from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils.six.moves.urllib.parse import unquote
def get_dvs_portgroup(self):
    pgroups = self.pgs
    pglist = []
    for pg in pgroups:
        trunk = False
        pvlan = False
        vlanInfo = pg.config.defaultPortConfig.vlan
        cl1 = vim.dvs.VmwareDistributedVirtualSwitch.TrunkVlanSpec
        cl2 = vim.dvs.VmwareDistributedVirtualSwitch.PvlanSpec
        vlan_id_list = []
        if isinstance(vlanInfo, cl1):
            trunk = True
            for item in vlanInfo.vlanId:
                if item.start == item.end:
                    vlan_id_list.append(str(item.start))
                else:
                    vlan_id_list.append(str(item.start) + '-' + str(item.end))
        elif isinstance(vlanInfo, cl2):
            pvlan = True
            vlan_id_list.append(str(vlanInfo.pvlanId))
        else:
            vlan_id_list.append(str(vlanInfo.vlanId))
        if self.cmp_vlans:
            if self.vlan_match(pg.config.uplink, self.module.params['show_uplink'], vlan_id_list):
                pglist.append(dict(name=unquote(pg.name), trunk=trunk, pvlan=pvlan, vlan_id=','.join(vlan_id_list), dvswitch=pg.config.distributedVirtualSwitch.name))
        else:
            pglist.append(dict(name=unquote(pg.name), trunk=trunk, pvlan=pvlan, vlan_id=','.join(vlan_id_list), dvswitch=pg.config.distributedVirtualSwitch.name))
    return pglist