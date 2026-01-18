from __future__ import absolute_import, division, print_function
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_LOCATION, scaleway_argument_spec, Scaleway
def stop_strategy(compute_api, wished_server):
    compute_api.module.debug('Starting stop strategy')
    query_results = find(compute_api=compute_api, wished_server=wished_server, per_page=1)
    changed = False
    if not query_results:
        if compute_api.module.check_mode:
            return (changed, {'status': 'A server would be created before being stopped.'})
        target_server = create_server(compute_api=compute_api, server=wished_server)
        changed = True
    else:
        target_server = query_results[0]
    compute_api.module.debug('stop_strategy: Servers are found.')
    if server_attributes_should_be_changed(compute_api=compute_api, target_server=target_server, wished_server=wished_server):
        changed = True
        if compute_api.module.check_mode:
            return (changed, {'status': 'Server %s attributes would be changed before stopping it.' % target_server['id']})
        target_server = server_change_attributes(compute_api=compute_api, target_server=target_server, wished_server=wished_server)
    wait_to_complete_state_transition(compute_api=compute_api, server=target_server)
    current_state = fetch_state(compute_api=compute_api, server=target_server)
    if current_state not in ('stopped',):
        compute_api.module.debug('stop_strategy: Server in state: %s' % current_state)
        changed = True
        if compute_api.module.check_mode:
            return (changed, {'status': 'Server %s would be stopped.' % target_server['id']})
        response = stop_server(compute_api=compute_api, server=target_server)
        compute_api.module.debug(response.json)
        compute_api.module.debug(response.ok)
        if not response.ok:
            msg = 'Error while stopping server [{0}: {1}]'.format(response.status_code, response.json)
            compute_api.module.fail_json(msg=msg)
    return (changed, target_server)