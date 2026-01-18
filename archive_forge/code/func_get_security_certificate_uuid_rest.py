from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_security_certificate_uuid_rest(self, name, type):
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 8):
        record, error = self.get_security_certificate_uuid_rest_98(name)
        message = 'certificate %s not found, retrying with common_name and type %s.' % (name, type)
    else:
        record, error = (None, None)
        message = 'name is not supported in 9.6 or 9.7, using common_name %s and type %s.' % (name, type)
    if not error and (not record):
        self.module.warn(message)
        record, error = self.get_security_certificate_uuid_rest_97(name, type)
    if not error and (not record):
        error = 'not found'
    if error:
        self.module.fail_json(msg='Error fetching security certificate info for %s of type: %s on %s: %s.' % (name, type, self.resource, error))
    return record['uuid']