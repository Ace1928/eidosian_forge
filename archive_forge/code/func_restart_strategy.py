from __future__ import absolute_import, division, print_function
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_LOCATION, scaleway_argument_spec, Scaleway
def restart_strategy(compute_api, wished_server):
    compute_api.module.debug('Starting restart strategy')
    changed = False
    query_results = find(compute_api=compute_api, wished_server=wished_server, per_page=1)
    if not query_results:
        changed = True
        if compute_api.module.check_mode:
            return (changed, {'status': 'A server would be created before being rebooted.'})
        target_server = create_server(compute_api=compute_api, server=wished_server)
    else:
        target_server = query_results[0]
    if server_attributes_should_be_changed(compute_api=compute_api, target_server=target_server, wished_server=wished_server):
        changed = True
        if compute_api.module.check_mode:
            return (changed, {'status': 'Server %s attributes would be changed before rebooting it.' % target_server['id']})
        target_server = server_change_attributes(compute_api=compute_api, target_server=target_server, wished_server=wished_server)
    changed = True
    if compute_api.module.check_mode:
        return (changed, {'status': 'Server %s would be rebooted.' % target_server['id']})
    wait_to_complete_state_transition(compute_api=compute_api, server=target_server)
    if fetch_state(compute_api=compute_api, server=target_server) in ('running',):
        response = restart_server(compute_api=compute_api, server=target_server)
        wait_to_complete_state_transition(compute_api=compute_api, server=target_server)
        if not response.ok:
            msg = 'Error while restarting server that was running [{0}: {1}].'.format(response.status_code, response.json)
            compute_api.module.fail_json(msg=msg)
    if fetch_state(compute_api=compute_api, server=target_server) in ('stopped',):
        response = restart_server(compute_api=compute_api, server=target_server)
        wait_to_complete_state_transition(compute_api=compute_api, server=target_server)
        if not response.ok:
            msg = 'Error while restarting server that was stopped [{0}: {1}].'.format(response.status_code, response.json)
            compute_api.module.fail_json(msg=msg)
    return (changed, target_server)