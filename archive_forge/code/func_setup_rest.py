from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def setup_rest(self):
    unsupported_rest_properties = ['identity_preserve', 'max_transfer_rate']
    host_options = self.parameters['peer_options'] if self.parameters.get('connection_type') == 'ontap_elementsw' else None
    rest_api = netapp_utils.OntapRestAPI(self.module, host_options=host_options)
    rtype = self.parameters.get('relationship_type')
    if rtype not in (None, 'extended_data_protection', 'restore'):
        unsupported_rest_properties.append('relationship_type')
    used_unsupported_rest_properties = [x for x in unsupported_rest_properties if x in self.parameters]
    ontap_97_options = ['create_destination', 'source_cluster', 'destination_cluster']
    partially_supported_rest_properties = [(property, (9, 7)) for property in ontap_97_options]
    partially_supported_rest_properties.extend([('schedule', (9, 11, 1)), ('identity_preservation', (9, 11, 1))])
    use_rest, error = rest_api.is_rest_supported_properties(self.parameters, used_unsupported_rest_properties, partially_supported_rest_properties, report_error=True)
    if error is not None:
        if 'relationship_type' in error:
            error = error.replace('relationship_type', 'relationship_type: %s' % rtype)
        if 'schedule' in error:
            error += ' - With REST use the policy option to define a schedule.'
        self.module.fail_json(msg=error)
    if not use_rest and any((x in self.parameters for x in ontap_97_options)):
        self.module.fail_json(msg='Error: %s' % rest_api.options_require_ontap_version(ontap_97_options, version='9.7', use_rest=use_rest))
    return (rest_api, use_rest)