from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_service_offering_id(self):
    service_offering = self.module.params.get('service_offering')
    service_offerings = self.query_api('listServiceOfferings')
    if service_offerings:
        if not service_offering:
            return service_offerings['serviceoffering'][0]['id']
        for s in service_offerings['serviceoffering']:
            if service_offering in [s['name'], s['id']]:
                return s['id']
    self.fail_json(msg="Service offering '%s' not found" % service_offering)