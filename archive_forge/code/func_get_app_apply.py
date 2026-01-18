from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_volume
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_app_apply(self):
    scope = self.na_helper.safe_get(self.parameters, ['san_application_template', 'scope'])
    app_current, error = self.rest_app.get_application_uuid()
    self.fail_on_error(error)
    if scope == 'lun' and app_current is None:
        self.module.fail_json(msg='Application not found: %s.  scope=%s.' % (self.na_helper.safe_get(self.parameters, ['san_application_template', 'name']), scope))
    return (scope, app_current)