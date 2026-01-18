from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_multiplier(self, unit):
    if not unit:
        return 1
    try:
        return netapp_utils.POW2_BYTE_MAP[unit[0].lower()]
    except KeyError:
        self.module.fail_json(msg='Error: unexpected unit in disk_size_with_unit: %s' % self.parameters['disk_size_with_unit'])