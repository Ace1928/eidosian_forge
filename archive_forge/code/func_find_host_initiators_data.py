from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def find_host_initiators_data(module, system, host, initiator_type):
    """
    Given a host object, find its initiators that match initiator_type.
    Only include desired initiator keys for each initiator.
    Return the filtered and edited host initiator list.
    """
    request = f'initiators?page=1&page_size=1000&host_id={host.id}'
    get_initiators_result = system.api.get(request, check_version=False)
    result_code = get_initiators_result.status_code
    if result_code != 200:
        msg = f'get initiators REST call failed. code: {result_code}'
        module.fail_json(msg=msg)
    host_initiators_by_type = [initiator for initiator in get_initiators_result.get_result() if initiator['type'] == initiator_type]
    if initiator_type == 'FC':
        include_key_list = ['address', 'address_long', 'host_id', 'port_key', 'targets', 'type']
    elif initiator_type == 'ISCSI':
        include_key_list = ['address', 'host_id', 'port_key', 'targets', 'type']
    else:
        msg = 'Cannot search for host initiator types other than FC and ISCSI'
        module.fail_json(msg=msg)
    host_initiators_by_type = edit_initiator_keys(host_initiators_by_type, include_key_list)
    return host_initiators_by_type