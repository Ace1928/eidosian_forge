from __future__ import absolute_import, division, print_function
import codecs
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text, to_bytes
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_owning_resource, rest_vserver
def get_ontap_subset_info_all(self, subset, default_fields, get_ontap_subset_info):
    """ Iteratively get all records for a subset """
    try:
        specified_subset = get_ontap_subset_info[subset]
    except KeyError:
        self.module.fail_json(msg='Specified subset %s is not found, supported subsets are %s' % (subset, list(get_ontap_subset_info.keys())))
    if 'api_call' not in specified_subset:
        specified_subset['api_call'] = subset
    subset_info = self.get_subset_info(specified_subset, default_fields)
    if subset_info is not None and isinstance(subset_info, dict) and ('_links' in subset_info):
        while subset_info['_links'].get('next'):
            next_api = subset_info['_links']['next']['href']
            gathered_subset_info = self.get_next_records(next_api.replace('/api', ''))
            subset_info['_links'] = gathered_subset_info['_links']
            subset_info['records'].extend(gathered_subset_info['records'])
        if subset_info.get('records') is not None:
            subset_info['num_records'] = len(subset_info['records'])
    return self.augment_subset_info(subset, subset_info)