from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
def get_host_initiators_list(self, host_details):
    """ Get the list of existing initiators in host"""
    existing_initiators = []
    if host_details.fc_host_initiators is not None:
        fc_len = len(host_details.fc_host_initiators)
        if fc_len > 0:
            for count in range(fc_len):
                " get initiator 'wwn' id"
                ini_id = host_details.fc_host_initiators.initiator_id[count]
                " update existing_initiators list with 'wwn' "
                existing_initiators.append(ini_id)
    if host_details.iscsi_host_initiators is not None:
        iscsi_len = len(host_details.iscsi_host_initiators)
        if iscsi_len > 0:
            for count in range(iscsi_len):
                " get initiator 'iqn' id"
                ini_id = host_details.iscsi_host_initiators.initiator_id[count]
                " update existing_initiators list with 'iqn' "
                existing_initiators.append(ini_id)
    return existing_initiators