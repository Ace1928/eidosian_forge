from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def report_error_in_modify(self, modify, context):
    if modify:
        if len(modify) > 1:
            tag = 'any of '
        else:
            tag = ''
        self.module.fail_json(msg='Error: modifying %s %s is not supported in %s' % (tag, str(modify), context))