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
def set_source_cluster_connection(self):
    """
        Setup ontap ZAPI or REST server connection for source hostname
        :return: None
        """
    self.src_rest_api = netapp_utils.OntapRestAPI(self.module, host_options=self.parameters['peer_options'])
    unsupported_rest_properties = ['identity_preserve', 'max_transfer_rate', 'schedule']
    rtype = self.parameters.get('relationship_type')
    if rtype not in (None, 'extended_data_protection', 'restore'):
        unsupported_rest_properties.append('relationship_type')
    used_unsupported_rest_properties = [x for x in unsupported_rest_properties if x in self.parameters]
    self.src_use_rest, error = self.src_rest_api.is_rest(used_unsupported_rest_properties)
    if error is not None:
        if 'relationship_type' in error:
            error = error.replace('relationship_type', 'relationship_type: %s' % rtype)
        self.module.fail_json(msg=error)
    if not self.src_use_rest:
        if not netapp_utils.has_netapp_lib():
            self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
        self.source_server = netapp_utils.setup_na_ontap_zapi(module=self.module, host_options=self.parameters['peer_options'])