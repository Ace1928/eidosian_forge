from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def remove_host_dict_for_non_adv(self, existing_host_dict, new_host_dict):
    """ Compares & remove new hosts from the existing ones and provide
            the remaining hosts for non advance host management

        :param existing_host_dict: All hosts params details which are
            associated with existing nfs which to be modified
        :type existing_host_dict: dict
        :param new_host_dict: All hosts param details which are to be removed
        :type new_host_dict: dict
        :return: existing hosts params details from which given new hosts are
            removed
        :rtype: dict
        """
    modify_host_dict = {}
    for host_access_key in existing_host_dict:
        LOG.debug('Checking remove host for param: %s', host_access_key)
        existing_host_str = existing_host_dict[host_access_key]
        existing_host_list = self.convert_host_str_to_list(existing_host_str)
        new_host_str = new_host_dict[host_access_key]
        new_host_list = self.convert_host_str_to_list(new_host_str)
        if not new_host_list:
            LOG.debug('Nothing to remove as no host given')
            continue
        if len(new_host_list) > len(set(new_host_list)):
            msg = 'Duplicate host given: %s in host param: %s' % (new_host_list, host_access_key)
            LOG.error(msg)
            self.module.fail_json(msg=msg)
        if new_host_list and (not existing_host_list):
            LOG.debug('Existing list is already empty, so nothing to remove')
            continue
        actual_to_remove = list(set(new_host_list) & set(existing_host_list))
        if not actual_to_remove:
            continue
        final_host_list = list(set(existing_host_list) - set(actual_to_remove))
        modify_host_dict[host_access_key] = ','.join((str(v) for v in final_host_list))
    return modify_host_dict