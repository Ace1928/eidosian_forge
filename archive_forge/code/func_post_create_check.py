from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def post_create_check(self, sd_id):
    storage_domain = self._service.service(sd_id).get()
    dc_name = self.param('data_center')
    if not dc_name:
        dc_name = self._find_attached_datacenter_name(storage_domain.name)
    self._service = self._attached_sds_service(dc_name)
    attached_sd_service = self._service.service(storage_domain.id)
    if get_entity(attached_sd_service) is None:
        self._service.add(otypes.StorageDomain(id=storage_domain.id))
        self.changed = True
        wait(service=attached_sd_service, condition=lambda sd: sd.status == sdstate.ACTIVE, wait=self.param('wait'), timeout=self.param('timeout'))