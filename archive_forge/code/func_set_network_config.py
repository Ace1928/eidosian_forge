from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
def set_network_config(self, network_object):
    """
        set network configuration
        """
    try:
        self.sfe.set_network_config(network=network_object)
    except (sf_ApiConnectionError, sf_ApiServerError) as exception_object:
        self.module.fail_json(msg='Error  setting network config for node %s' % to_native(exception_object), exception=traceback.format_exc())