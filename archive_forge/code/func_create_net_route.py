from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_net_route(self, current=None, fail=True):
    """
        Creates a new Route
        """
    if current is None:
        current = self.parameters
    if self.use_rest:
        api = 'network/ip/routes'
        body = {'gateway': current['gateway']}
        dest = current['destination']
        if isinstance(dest, dict):
            body['destination'] = dest
        else:
            dest = current['destination'].split('/')
            body['destination'] = {'address': dest[0], 'netmask': dest[1]}
        if current.get('vserver') is not None:
            body['svm.name'] = current['vserver']
        if current.get('metric') is not None:
            body['metric'] = current['metric']
        __, error = rest_generic.post_async(self.rest_api, api, body)
    else:
        route_obj = netapp_utils.zapi.NaElement('net-routes-create')
        route_obj.add_new_child('destination', current['destination'])
        route_obj.add_new_child('gateway', current['gateway'])
        metric = current.get('metric')
        if metric is not None:
            route_obj.add_new_child('metric', str(metric))
        try:
            self.server.invoke_successfully(route_obj, True)
            error = None
        except netapp_utils.zapi.NaApiError as exc:
            error = self.sanitize_exception('create', exc)
    if error:
        error = 'Error creating net route: %s' % error
        if fail:
            self.module.fail_json(msg=error)
    return error