from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def recreate_net_route(self, current):
    """
        Modify a net route
        Since we cannot modify a route, we are deleting the existing route, and creating a new one.
        """
    self.delete_net_route(current)
    if current.get('metric') is not None and self.parameters.get('metric') is None:
        self.parameters['metric'] = current['metric']
    error = self.create_net_route(fail=False)
    if error:
        self.create_net_route(current)
        self.module.fail_json(msg='Error modifying net route: %s' % error, exception=traceback.format_exc())