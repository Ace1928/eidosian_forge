from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_volume
def rest_offline_volume(self, current):
    """
        Offline the volume using REST PATCH method.
        """
    uuid = current.get('uuid')
    if uuid is None:
        error = 'Error, no uuid in current: %s' % str(current)
        self.na_helper.fail_on_error(error)
    body = dict(state='offline')
    return self.patch_volume_rest(uuid, body)