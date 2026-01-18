from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleActionFail
from ansible.module_utils.connection import Connection
from ansible.module_utils.six.moves.urllib.parse import quote_plus
from ansible.plugins.action import ActionBase
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.splunk.es.plugins.module_utils.splunk import (
from ansible_collections.splunk.es.plugins.modules.splunk_data_inputs_network import DOCUMENTATION
def request_by_path(self, conn_request, protocol, datatype=None, name=None, req_type='get', payload=None):
    query_dict = None
    url = ''
    if protocol == 'tcp':
        if not datatype:
            raise AnsibleActionFail('No datatype specified for TCP input')
        if not name or (req_type == 'post_create' and datatype != 'ssl'):
            name = ''
        url = '{0}/{1}/{2}/{3}'.format(self.api_object, protocol, datatype, quote_plus(str(name)))
        if url[-1] == '/':
            url = url[:-1]
    elif protocol == 'udp':
        if datatype:
            raise AnsibleActionFail('Datatype specified for UDP input')
        if not name or req_type == 'post_create':
            name = ''
        url = '{0}/{1}/{2}'.format(self.api_object, protocol, quote_plus(str(name)))
        if url[-1] == '/':
            url = url[:-1]
    else:
        raise AnsibleActionFail("Incompatible protocol specified. Please specify 'tcp' or 'udp'")
    if req_type == 'get':
        query_dict = conn_request.get_by_path(url)
    elif req_type == 'post_create':
        query_dict = conn_request.create_update(url, data=payload)
    elif req_type == 'post_update':
        payload.pop('name')
        query_dict = conn_request.create_update(url, data=payload)
    elif req_type == 'delete':
        query_dict = conn_request.delete_by_path(url)
    return query_dict