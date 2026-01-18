from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def gather_secondarys_ids(self, mdm, cluster_details):
    """ Prepare a list of secondary MDMs for switch cluster mode
            operation"""
    secondarys = []
    for node in mdm:
        name_or_id = node['mdm_name'] if node['mdm_name'] else node['mdm_id']
        if node['mdm_type'] == 'Secondary' and node['mdm_id'] is not None:
            mdm_details = self.is_mdm_name_id_exists(mdm_id=node['mdm_id'], cluster_details=cluster_details)
            if mdm_details is None:
                err_msg = self.not_exist_msg.format(name_or_id)
                self.module.fail_json(msg=err_msg)
            secondarys.append(node['mdm_id'])
        elif node['mdm_type'] == 'Secondary' and node['mdm_name'] is not None:
            mdm_details = self.is_mdm_name_id_exists(mdm_name=node['mdm_name'], cluster_details=cluster_details)
            if mdm_details is None:
                err_msg = self.not_exist_msg.format(name_or_id)
                self.module.fail_json(msg=err_msg)
            else:
                secondarys.append(mdm_details['id'])
    return secondarys