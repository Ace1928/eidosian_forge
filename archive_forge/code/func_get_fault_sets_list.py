from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def get_fault_sets_list(self, filter_dict=None):
    """ Get the list of fault sets on a given PowerFlex storage
            system """
    try:
        LOG.info('Getting fault set list ')
        filter_pd = []
        if filter_dict:
            if 'protectionDomainName' in filter_dict.keys():
                filter_pd = filter_dict['protectionDomainName']
                del filter_dict['protectionDomainName']
            fault_sets = self.powerflex_conn.fault_set.get(filter_fields=filter_dict)
        else:
            fault_sets = self.powerflex_conn.fault_set.get()
        fault_set_final = []
        if fault_sets:
            for fault_set in fault_sets:
                fault_set['protectionDomainName'] = Configuration(self.powerflex_conn, self.module).get_protection_domain(protection_domain_id=fault_set['protectionDomainId'])['name']
                fault_set['SDS'] = Configuration(self.powerflex_conn, self.module).get_associated_sds(fault_set_id=fault_set['id'])
                fault_set_final.append(fault_set)
        fault_sets = []
        for fault_set in fault_set_final:
            if fault_set['protectionDomainName'] in filter_pd:
                fault_sets.append(fault_set)
        if len(filter_pd) != 0:
            return result_list(fault_sets)
        return result_list(fault_set_final)
    except Exception as e:
        msg = 'Get fault set list from powerflex array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)