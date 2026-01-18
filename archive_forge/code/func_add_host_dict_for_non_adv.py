from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def add_host_dict_for_non_adv(self, existing_host_dict, new_host_dict):
    """ Compares & adds up new hosts with the existing ones and provide
            the final consolidated hosts for non advance host management

        :param existing_host_dict: All hosts params details which are
            associated with existing nfs which to be modified
        :type existing_host_dict: dict
        :param new_host_dict: All hosts param details which are to be added
        :type new_host_dict: dict
        :return: consolidated hosts params details which contains newly added
            hosts along with the existing ones
        :rtype: dict
        """
    modify_host_dict = {}
    for host_access_key in existing_host_dict:
        LOG.debug('Checking add host for param: %s', host_access_key)
        existing_host_str = existing_host_dict[host_access_key]
        existing_host_list = self.convert_host_str_to_list(existing_host_str)
        new_host_str = new_host_dict[host_access_key]
        new_host_list = self.convert_host_str_to_list(new_host_str)
        if not new_host_list:
            LOG.debug('Nothing to add as no host given')
            continue
        if new_host_list and (not existing_host_list):
            LOG.debug('Existing nfs host key: %s is empty, so lets add new host given value as it is', host_access_key)
            modify_host_dict[host_access_key] = new_host_str
            continue
        actual_to_add = list(set(new_host_list) - set(existing_host_list))
        if not actual_to_add:
            LOG.debug('All host given to be added is already added')
            continue
        actual_to_add.extend(existing_host_list)
        modify_host_dict[host_access_key] = ','.join((str(v) for v in actual_to_add))
    return modify_host_dict