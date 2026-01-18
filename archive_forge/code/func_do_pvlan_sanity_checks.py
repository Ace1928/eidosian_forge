from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def do_pvlan_sanity_checks(self):
    """Do sanity checks for primary and secondary PVLANs"""
    for primary_vlan in self.primary_pvlans:
        count = 0
        primary_pvlan_id = self.get_primary_pvlan_option(primary_vlan)
        for primary_vlan_2 in self.primary_pvlans:
            primary_pvlan_id_2 = self.get_primary_pvlan_option(primary_vlan_2)
            if primary_pvlan_id == primary_pvlan_id_2:
                count += 1
        if count > 1:
            self.module.fail_json(msg="The primary PVLAN ID '%s' must be unique!" % primary_pvlan_id)
    if self.secondary_pvlans:
        for secondary_pvlan in self.secondary_pvlans:
            count = 0
            result = self.get_secondary_pvlan_options(secondary_pvlan)
            for secondary_pvlan_2 in self.secondary_pvlans:
                result_2 = self.get_secondary_pvlan_options(secondary_pvlan_2)
                if result[0] == result_2[0]:
                    count += 1
            if count > 1:
                self.module.fail_json(msg="The secondary PVLAN ID '%s' must be unique!" % result[0])
        for primary_vlan in self.primary_pvlans:
            primary_pvlan_id = self.get_primary_pvlan_option(primary_vlan)
            for secondary_pvlan in self.secondary_pvlans:
                result = self.get_secondary_pvlan_options(secondary_pvlan)
                if primary_pvlan_id == result[0]:
                    self.module.fail_json(msg="The secondary PVLAN ID '%s' is already used as a primary PVLAN!" % result[0])
        for secondary_pvlan in self.secondary_pvlans:
            primary_pvlan_found = False
            result = self.get_secondary_pvlan_options(secondary_pvlan)
            for primary_vlan in self.primary_pvlans:
                primary_pvlan_id = self.get_primary_pvlan_option(primary_vlan)
                if result[1] == primary_pvlan_id:
                    primary_pvlan_found = True
                    break
            if not primary_pvlan_found:
                self.module.fail_json(msg="The primary PVLAN ID '%s' isn't defined for the secondary PVLAN ID '%s'!" % (result[1], result[0]))