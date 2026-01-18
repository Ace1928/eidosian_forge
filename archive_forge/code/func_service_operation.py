from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
def service_operation(module, auth, service_id=None, owner_id=None, group_id=None, permissions=None, role=None, cardinality=None, force=None, wait=False, wait_timeout=None, service=None):
    changed = False
    if not service:
        service = get_service_by_id(module, auth, service_id)
    else:
        service_id = service['ID']
    if not service:
        module.fail_json(msg='There is no service with id: ' + str(service_id))
    if owner_id:
        if check_change_service_owner(module, service, owner_id):
            if not module.check_mode:
                change_service_owner(module, auth, service_id, owner_id)
            changed = True
    if group_id:
        if check_change_service_group(module, service, group_id):
            if not module.check_mode:
                change_service_group(module, auth, service_id, group_id)
            changed = True
    if permissions:
        if check_change_service_permissions(module, service, permissions):
            if not module.check_mode:
                change_service_permissions(module, auth, service_id, permissions)
            changed = True
    if role:
        if check_change_role_cardinality(module, service, role, cardinality):
            if not module.check_mode:
                change_role_cardinality(module, auth, service_id, role, cardinality, force)
            changed = True
    if wait and (not module.check_mode):
        service = wait_for_service_to_become_ready(module, auth, service_id, wait_timeout)
    if changed:
        service = get_service_by_id(module, auth, service_id)
    service_info = get_service_info(module, auth, service)
    service_info['changed'] = changed
    return service_info