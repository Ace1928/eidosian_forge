from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def check_mvip_connection(self):
    """
            Check connection to MVIP

            :return: true if connection was successful, false otherwise.
            :rtype: bool
        """
    try:
        test = self.elem.test_connect_mvip(mvip=self.parameters['mvip'])
        return test.details.connected
    except Exception as e:
        self.msg += 'Error checking connection to MVIP: %s' % to_native(e)
        return False