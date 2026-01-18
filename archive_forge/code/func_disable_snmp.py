from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def disable_snmp(self):
    """
        disable snmp feature
        """
    try:
        self.sfe.disable_snmp()
    except Exception as exception_object:
        self.module.fail_json(msg='Error disabling snmp feature %s' % to_native(exception_object), exception=traceback.format_exc())