from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils.six.moves.urllib.parse import unquote
def get_vlan_info(self, vlan_obj=None):
    """
        Return vlan information from given object
        Args:
            vlan_obj: vlan managed object
        Returns: Dict of vlan details of the specific object
        """
    vdret = dict()
    if not vlan_obj:
        return vdret
    if isinstance(vlan_obj, vim.dvs.VmwareDistributedVirtualSwitch.TrunkVlanSpec):
        vlan_id_list = []
        for vli in vlan_obj.vlanId:
            if vli.start == vli.end:
                vlan_id_list.append(str(vli.start))
            else:
                vlan_id_list.append(str(vli.start) + '-' + str(vli.end))
        vdret = dict(trunk=True, pvlan=False, vlan_id=vlan_id_list)
    elif isinstance(vlan_obj, vim.dvs.VmwareDistributedVirtualSwitch.PvlanSpec):
        vdret = dict(trunk=False, pvlan=True, vlan_id=str(vlan_obj.pvlanId))
    else:
        vdret = dict(trunk=False, pvlan=False, vlan_id=str(vlan_obj.vlanId))
    return vdret