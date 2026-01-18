from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
def secure_erase(self, drives=None):
    """
        Secure Erase any residual data existing on a drive
        """
    try:
        self.sfe.secure_erase_drives(drives)
    except Exception as exception_object:
        self.module.fail_json(msg='Error cleaning data from drive%s: %s: %s' % ('s' if len(drives) > 1 else '', str(drives), to_native(exception_object)), exception=traceback.format_exc())