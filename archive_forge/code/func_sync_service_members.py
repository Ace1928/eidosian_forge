from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import copy
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (ConfigProxy, get_nitro_client, netscaler_common_arguments,
def sync_service_members(client, module):
    log('sync_service_members')
    configured_service_members = get_configured_service_members(client, module)
    actual_service_members = get_actual_service_members(client, module)
    skip_add = []
    skip_delete = []
    for configured_index, configured_service in enumerate(configured_service_members):
        for actual_index, actual_service in enumerate(actual_service_members):
            if configured_service.has_equal_attributes(actual_service):
                skip_add.append(configured_index)
                skip_delete.append(actual_index)
    for actual_index, actual_service in enumerate(actual_service_members):
        if actual_index in skip_delete:
            log('Skipping actual delete at index %s' % actual_index)
            continue
        if all([hasattr(actual_service, 'ip'), actual_service.ip is not None, hasattr(actual_service, 'servername'), actual_service.servername is not None]):
            actual_service.ip = None
        actual_service.servicegroupname = module.params['servicegroupname']
        servicegroup_servicegroupmember_binding.delete(client, actual_service)
    for configured_index, configured_service in enumerate(configured_service_members):
        if configured_index in skip_add:
            log('Skipping configured add at index %s' % configured_index)
            continue
        configured_service.add()