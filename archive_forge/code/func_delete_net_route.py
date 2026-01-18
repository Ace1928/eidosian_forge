from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_net_route(self, current):
    """
        Deletes a given Route
        """
    if self.use_rest:
        uuid = current['uuid']
        api = 'network/ip/routes'
        dummy, error = rest_generic.delete_async(self.rest_api, api, uuid)
        if error:
            self.module.fail_json(msg='Error deleting net route - %s' % error)
    else:
        route_obj = netapp_utils.zapi.NaElement('net-routes-destroy')
        route_obj.add_new_child('destination', current['destination'])
        route_obj.add_new_child('gateway', current['gateway'])
        try:
            self.server.invoke_successfully(route_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error deleting net route: %s' % to_native(error), exception=traceback.format_exc())