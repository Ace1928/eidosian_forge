from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def reconfig_kmip_standard_kp(self, kmip_cluster_servers, kms_info_list, proxy_user_config_dict):
    changed = False
    if len(kms_info_list) != 0:
        for kms_info in kms_info_list:
            existing_kmip = None
            for kmip_server in kmip_cluster_servers:
                if kmip_server.name == kms_info.get('kms_name'):
                    existing_kmip = kmip_server
            if existing_kmip is not None:
                if kms_info.get('remove_kms'):
                    self.remove_kms_server(self.key_provider_id, kms_info.get('kms_name'))
                    kms_changed = True
                else:
                    kms_changed = self.change_kmip_in_standard_kp(existing_kmip, kms_info, proxy_user_config_dict)
            else:
                if kms_info.get('remove_kms'):
                    self.module.fail_json(msg="Not find named KMS server to remove in the key provider cluster '%s'" % self.key_provider_id.id)
                self.add_kmip_to_standard_kp(kms_info, proxy_user_config_dict)
                kms_changed = True
            if kms_changed:
                changed = True
    for kmip_server in kmip_cluster_servers:
        kms_changed = self.change_kmip_in_standard_kp(kmip_server, kms_info=None, proxy_user_config_dict=proxy_user_config_dict)
        if kms_changed:
            changed = True
    return changed