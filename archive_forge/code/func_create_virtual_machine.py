from __future__ import absolute_import, division, print_function
import re
import uuid
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import xrange
from ansible.module_utils.common.text.converters import to_native
def create_virtual_machine(module, profitbricks):
    """
    Create new virtual machine

    module : AnsibleModule object
    community.general.profitbricks: authenticated profitbricks object

    Returns:
        True if a new virtual machine was created, false otherwise
    """
    datacenter = module.params.get('datacenter')
    name = module.params.get('name')
    auto_increment = module.params.get('auto_increment')
    count = module.params.get('count')
    lan = module.params.get('lan')
    wait_timeout = module.params.get('wait_timeout')
    failed = True
    datacenter_found = False
    virtual_machines = []
    virtual_machine_ids = []
    datacenter_list = profitbricks.list_datacenters()
    datacenter_id = _get_datacenter_id(datacenter_list, datacenter)
    if datacenter_id:
        datacenter_found = True
    if not datacenter_found:
        datacenter_response = _create_datacenter(module, profitbricks)
        datacenter_id = datacenter_response['id']
        _wait_for_completion(profitbricks, datacenter_response, wait_timeout, 'create_virtual_machine')
    if auto_increment:
        numbers = set()
        count_offset = 1
        try:
            name % 0
        except TypeError as e:
            if e.message.startswith('not all'):
                name = '%s%%d' % name
            else:
                module.fail_json(msg=e.message, exception=traceback.format_exc())
        number_range = xrange(count_offset, count_offset + count + len(numbers))
        available_numbers = list(set(number_range).difference(numbers))
        names = []
        numbers_to_use = available_numbers[:count]
        for number in numbers_to_use:
            names.append(name % number)
    else:
        names = [name]
    server_list = profitbricks.list_servers(datacenter_id)
    for name in names:
        if _get_server_id(server_list, name):
            continue
        create_response = _create_machine(module, profitbricks, str(datacenter_id), name)
        nics = profitbricks.list_nics(datacenter_id, create_response['id'])
        for n in nics['items']:
            if lan == n['properties']['lan']:
                create_response.update({'public_ip': n['properties']['ips'][0]})
        virtual_machines.append(create_response)
    failed = False
    results = {'failed': failed, 'machines': virtual_machines, 'action': 'create', 'instance_ids': {'instances': [i['id'] for i in virtual_machines]}}
    return results