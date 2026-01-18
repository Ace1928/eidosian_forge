from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_conn_array_dict(module, array):
    conn_array_info = {}
    api_version = array._list_available_rest_versions()
    if FC_REPL_API_VERSION not in api_version:
        carrays = array.list_array_connections()
        for carray in range(0, len(carrays)):
            arrayname = carrays[carray]['array_name']
            conn_array_info[arrayname] = {'array_id': carrays[carray]['id'], 'throttled': carrays[carray]['throttled'], 'version': carrays[carray]['version'], 'type': carrays[carray]['type'], 'mgmt_ip': carrays[carray]['management_address'], 'repl_ip': carrays[carray]['replication_address']}
            if P53_API_VERSION in api_version:
                conn_array_info[arrayname]['status'] = carrays[carray]['status']
            else:
                conn_array_info[arrayname]['connected'] = carrays[carray]['connected']
        throttles = array.list_array_connections(throttle=True)
        for throttle in range(0, len(throttles)):
            arrayname = throttles[throttle]['array_name']
            if conn_array_info[arrayname]['throttled']:
                conn_array_info[arrayname]['throttling'] = {'default_limit': throttles[throttle]['default_limit'], 'window_limit': throttles[throttle]['window_limit'], 'window': throttles[throttle]['window']}
    else:
        arrayv6 = get_array(module)
        carrays = list(arrayv6.get_array_connections().items)
        for carray in range(0, len(carrays)):
            arrayname = carrays[carray].name
            conn_array_info[arrayname] = {'array_id': carrays[carray].id, 'version': getattr(carrays[carray], 'version', None), 'status': carrays[carray].status, 'type': carrays[carray].type, 'mgmt_ip': getattr(carrays[carray], 'management_address', '-'), 'repl_ip': getattr(carrays[carray], 'replication_addresses', '-'), 'transport': carrays[carray].replication_transport}
            if bool(carrays[carray].throttle.to_dict()):
                conn_array_info[arrayname]['throttled'] = True
                conn_array_info[arrayname]['throttling'] = {}
                try:
                    if bool(carrays[carray].throttle.window):
                        conn_array_info[arrayname]['throttling']['window'] = carrays[carray].throttle.window.to_dict()
                except AttributeError:
                    pass
                try:
                    if bool(carrays[carray].throttle.default_limit):
                        conn_array_info[arrayname]['throttling']['default_limit'] = carrays[carray].throttle.default_limit
                except AttributeError:
                    pass
                try:
                    if bool(carrays[carray].throttle.window_limit):
                        conn_array_info[arrayname]['throttling']['window_limit'] = carrays[carray].throttle.window_limit
                except AttributeError:
                    pass
            else:
                conn_array_info[arrayname]['throttled'] = False
    return conn_array_info