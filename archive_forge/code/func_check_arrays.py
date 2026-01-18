from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def check_arrays(module, array):
    """Check if array name provided are sync-replicated"""
    good_arrays = []
    good_arrays.append(array.get()['array_name'])
    connected_arrays = array.list_array_connections()
    for arr in range(0, len(connected_arrays)):
        if connected_arrays[arr]['type'] == 'sync-replication':
            good_arrays.append(connected_arrays[arr]['array_name'])
    if module.params['failover'] is not None:
        if module.params['failover'] == ['auto']:
            failover_array = []
        else:
            failover_array = module.params['failover']
        if failover_array != []:
            for arr in range(0, len(failover_array)):
                if failover_array[arr] not in good_arrays:
                    module.fail_json(msg='Failover array {0} is not valid.'.format(failover_array[arr]))
    if module.params['stretch'] is not None:
        if module.params['stretch'] not in good_arrays:
            module.fail_json(msg='Stretch: Array {0} is not connected.'.format(module.params['stretch']))
    return None