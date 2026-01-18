from __future__ import absolute_import, division, print_function
import re
import uuid
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import xrange
from ansible.module_utils.common.text.converters import to_native
def startstop_machine(module, profitbricks, state):
    """
    Starts or Stops a virtual machine.

    module : AnsibleModule object
    community.general.profitbricks: authenticated profitbricks object.

    Returns:
        True when the servers process the action successfully, false otherwise.
    """
    if not isinstance(module.params.get('instance_ids'), list) or len(module.params.get('instance_ids')) < 1:
        module.fail_json(msg='instance_ids should be a list of virtual machine ids or names, aborting')
    wait = module.params.get('wait')
    wait_timeout = module.params.get('wait_timeout')
    changed = False
    datacenter = module.params.get('datacenter')
    instance_ids = module.params.get('instance_ids')
    datacenter_list = profitbricks.list_datacenters()
    datacenter_id = _get_datacenter_id(datacenter_list, datacenter)
    if not datacenter_id:
        module.fail_json(msg="Virtual data center '%s' not found." % str(datacenter))
    server_list = profitbricks.list_servers(datacenter_id)
    for instance in instance_ids:
        server_id = _get_server_id(server_list, instance)
        if server_id:
            _startstop_machine(module, profitbricks, datacenter_id, server_id)
            changed = True
    if wait:
        wait_timeout = time.time() + wait_timeout
        while wait_timeout > time.time():
            matched_instances = []
            for res in profitbricks.list_servers(datacenter_id)['items']:
                if state == 'running':
                    if res['properties']['vmState'].lower() == state:
                        matched_instances.append(res)
                elif state == 'stopped':
                    if res['properties']['vmState'].lower() == 'shutoff':
                        matched_instances.append(res)
            if len(matched_instances) < len(instance_ids):
                time.sleep(5)
            else:
                break
        if wait_timeout <= time.time():
            module.fail_json(msg='wait for virtual machine state timeout on %s' % time.asctime())
    return changed