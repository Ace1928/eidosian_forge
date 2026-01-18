from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_image_get_rest(self, what, fail_on_error=True):
    """return field information for:
            - nodes if what == versions
            - validation_results if what == validation_results
            - state if what == state
            - any other field if what is a valid field name
           call fail_json when there is an error and fail_on_error is True
           return a tuple (info, error) when fail_on_error is False
           return info when fail_on_error is Trie
        """
    api = 'cluster/software'
    field = 'nodes' if what == 'versions' else what
    record, error = rest_generic.get_one_record(self.rest_api, api, fields=field)
    optional_fields = ['validation_results']
    info, error_msg = (None, None)
    if error or not record:
        if error or field not in optional_fields:
            error_msg = 'Error fetching software information for %s: %s' % (field, error or 'no record calling %s' % api)
    elif what == 'versions' and 'nodes' in record:
        nodes = self.parameters.get('nodes')
        if nodes:
            known_nodes = [node['name'] for node in record['nodes']]
            unknown_nodes = [node for node in nodes if node not in known_nodes]
            if unknown_nodes:
                error_msg = 'Error: node%s not found in cluster: %s.' % ('s' if len(unknown_nodes) > 1 else '', ', '.join(unknown_nodes))
        info = [(node['name'], node['version']) for node in record['nodes'] if nodes is None or node['name'] in nodes]
    elif field in record:
        info = record[field]
    elif field not in optional_fields:
        error_msg = 'Unexpected results for what: %s, record: %s' % (what, record)
    if fail_on_error and error_msg:
        self.module.fail_json(msg=error_msg)
    return info if fail_on_error else (info, error_msg)