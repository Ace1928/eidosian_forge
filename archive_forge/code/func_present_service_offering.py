from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_service_offering(self):
    service_offering = self.get_service_offering()
    if not service_offering:
        service_offering = self._create_offering(service_offering)
    else:
        service_offering = self._update_offering(service_offering)
    return service_offering