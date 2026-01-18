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
def set_endpoints(self):
    for location in ('source', 'destination'):
        endpoint = '%s_endpoint' % location
        self.parameters[endpoint] = {}
        for old, new in (('path', 'path'), ('cluster', 'cluster')):
            value = self.parameters.get('%s_%s' % (location, old))
            if value is not None:
                self.parameters[endpoint][new] = value