from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_net_route(self, params=None):
    """
        Checks to see if a route exist or not
        :return: NaElement object if a route exists, None otherwise
        """
    if params is None:
        params = self.parameters
    if self.use_rest:
        api = 'network/ip/routes'
        fields = 'destination,gateway,svm,scope'
        if self.parameters.get('metric') is not None:
            fields += ',metric'
        query = {'destination.address': params['destination'].split('/')[0], 'gateway': params['gateway']}
        if params.get('vserver') is None:
            query['scope'] = 'cluster'
        else:
            query['scope'] = 'svm'
            query['svm.name'] = params['vserver']
        record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
        if error:
            self.module.fail_json(msg='Error fetching net route: %s' % error)
        if record and 'metric' not in record:
            record['metric'] = None
        return record
    else:
        route_obj = netapp_utils.zapi.NaElement('net-routes-get')
        for attr in ('destination', 'gateway'):
            route_obj.add_new_child(attr, params[attr])
        try:
            result = self.server.invoke_successfully(route_obj, True)
        except netapp_utils.zapi.NaApiError as exc:
            error = self.sanitize_exception('get', exc)
            if error is None:
                return None
            self.module.fail_json(msg='Error fetching net route: %s' % error, exception=traceback.format_exc())
        if result.get_child_by_name('attributes') is not None:
            route_info = result.get_child_by_name('attributes').get_child_by_name('net-vs-routes-info')
            return {'destination': route_info.get_child_content('destination'), 'gateway': route_info.get_child_content('gateway'), 'metric': int(route_info.get_child_content('metric'))}
        return None