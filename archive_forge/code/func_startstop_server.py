from __future__ import absolute_import, division, print_function
import os
import time
from ansible.module_utils.six.moves import xrange
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def startstop_server(module, oneandone_conn):
    """
    Starts or Stops a server.

    module : AnsibleModule object
    oneandone_conn: authenticated oneandone object.

    Returns a dictionary with a 'changed' attribute indicating whether
    anything has changed for the server as a result of this function
    being run, and a 'server' attribute with basic information for
    the server.
    """
    state = module.params.get('state')
    server_id = module.params.get('server')
    wait = module.params.get('wait')
    wait_timeout = module.params.get('wait_timeout')
    wait_interval = module.params.get('wait_interval')
    changed = False
    server = get_server(oneandone_conn, server_id, True)
    if server:
        try:
            if state == 'stopped' and server['status']['state'] == 'POWERED_ON':
                _check_mode(module, True)
                oneandone_conn.modify_server_status(server_id=server['id'], action='POWER_OFF', method='SOFTWARE')
            elif state == 'running' and server['status']['state'] == 'POWERED_OFF':
                _check_mode(module, True)
                oneandone_conn.modify_server_status(server_id=server['id'], action='POWER_ON', method='SOFTWARE')
        except Exception as ex:
            module.fail_json(msg='failed to set server %s to state %s: %s' % (server_id, state, str(ex)))
        _check_mode(module, False)
        if wait:
            operation_completed = False
            wait_timeout = time.time() + wait_timeout
            while wait_timeout > time.time():
                time.sleep(wait_interval)
                server = oneandone_conn.get_server(server['id'])
                server_state = server['status']['state']
                if state == 'stopped' and server_state == 'POWERED_OFF':
                    operation_completed = True
                    break
                if state == 'running' and server_state == 'POWERED_ON':
                    operation_completed = True
                    break
            if not operation_completed:
                module.fail_json(msg='Timeout waiting for server %s to get to state %s' % (server_id, state))
        changed = True
        server = _insert_network_data(server)
    _check_mode(module, False)
    return (changed, server)