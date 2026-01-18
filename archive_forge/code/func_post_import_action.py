from __future__ import (absolute_import, division, print_function)
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def post_import_action(self, entity):
    self._service = self._connection.system_service().templates_service()