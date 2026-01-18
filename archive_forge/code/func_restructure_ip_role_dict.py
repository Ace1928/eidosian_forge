from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
def restructure_ip_role_dict(self, sds_ip_list):
    """Restructure IP role dict
            :param sds_ip_list: List of one or more IP addresses and
                                their roles
            :type sds_ip_list: list[dict]
            :return: List of one or more IP addresses and their roles
            :rtype: list[dict]
        """
    new_sds_ip_list = []
    for item in sds_ip_list:
        new_sds_ip_list.append({'SdsIp': item})
    return new_sds_ip_list