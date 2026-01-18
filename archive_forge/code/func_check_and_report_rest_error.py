from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def check_and_report_rest_error(self, error, action, where):
    if error:
        if 'job reported error:' in error and "entry doesn't exist" in error:
            self.module.warn('Ignoring job status, assuming success - Issue #45.')
            return
        self.module.fail_json(msg='Error %s vserver peer relationship on %s: %s' % (action, where, error))