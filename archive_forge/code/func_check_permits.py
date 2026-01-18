from __future__ import (absolute_import, division, print_function)
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
from ansible.module_utils.basic import AnsibleModule
import traceback
def check_permits():
    if self.param('permits'):
        if 'login' not in self.param('permits'):
            self.param('permits').append('login')
        permits_service = self._service.service(entity.id).permits_service()
        current = [er.name for er in permits_service.list()]
        passed = self.param('permits')
        if not sorted(current) == sorted(passed):
            if self._module.check_mode:
                return False
            for permit in permits_service.list():
                permits_service.permit_service(permit.id).remove()
            all_permits = self.get_all_permits()
            for new_permit in passed:
                permits_service.add(otypes.Permit(id=all_permits.get(new_permit)))
            return False
    return True