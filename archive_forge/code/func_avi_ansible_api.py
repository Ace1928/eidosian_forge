from __future__ import absolute_import, division, print_function
import os
import re
import logging
import sys
from copy import deepcopy
from ansible.module_utils.basic import env_fallback
def avi_ansible_api(module, obj_type, sensitive_fields):
    """
    This converts the Ansible module into AVI object and invokes APIs
    :param module: Ansible module
    :param obj_type: string representing Avi object type
    :param sensitive_fields: sensitive fields to be excluded for comparison
        purposes.
    Returns:
        success: module.exit_json with obj=avi object
        faliure: module.fail_json
    """
    api_creds = AviCredentials()
    api_creds.update_from_ansible_module(module)
    api_context = get_api_context(module, api_creds)
    if api_context:
        api = ApiSession.get_session(api_creds.controller, api_creds.username, password=api_creds.password, timeout=api_creds.timeout, tenant=api_creds.tenant, tenant_uuid=api_creds.tenant_uuid, token=api_context['csrftoken'], port=api_creds.port, session_id=api_context['session_id'], csrftoken=api_context['csrftoken'])
    else:
        api = ApiSession.get_session(api_creds.controller, api_creds.username, password=api_creds.password, timeout=api_creds.timeout, tenant=api_creds.tenant, tenant_uuid=api_creds.tenant_uuid, token=api_creds.token, port=api_creds.port)
    state = module.params['state']
    avi_update_method = module.params.get('avi_api_update_method', 'put')
    avi_patch_op = module.params.get('avi_api_patch_op', 'add')
    api_version = api_creds.api_version
    name = module.params.get('name', None)
    uuid = module.params.get('uuid', None)
    check_mode = module.check_mode
    if uuid and obj_type != 'cluster':
        obj_path = '%s/%s' % (obj_type, uuid)
    else:
        obj_path = '%s/' % obj_type
    obj = deepcopy(module.params)
    tenant = obj.pop('tenant', '')
    tenant_uuid = obj.pop('tenant_uuid', '')
    for k in POP_FIELDS:
        obj.pop(k, None)
    purge_optional_fields(obj, module)
    if 'obj_username' in obj:
        obj['username'] = obj['obj_username']
        obj.pop('obj_username')
    if 'obj_password' in obj:
        obj['password'] = obj['obj_password']
        obj.pop('obj_password')
    if 'full_name' not in obj and 'name' in obj and (obj_type == 'user'):
        obj['full_name'] = obj['name']
        obj['name'] = obj['username']
    log.info('passed object %s ', obj)
    if uuid:
        try:
            existing_obj = api.get(obj_path, tenant=tenant, tenant_uuid=tenant_uuid, params={'include_refs': '', 'include_name': ''}, api_version=api_version)
            existing_obj = existing_obj.json()
        except ObjectNotFound:
            existing_obj = None
    elif name:
        params = {'include_refs': '', 'include_name': ''}
        if obj.get('cloud_ref', None):
            cloud = obj['cloud_ref'].split('name=')[1]
            params['cloud_ref.name'] = cloud
        existing_obj = api.get_object_by_name(obj_type, name, tenant=tenant, tenant_uuid=tenant_uuid, params=params, api_version=api_version)
        if existing_obj and 'tenant_ref' in obj and ('tenant_ref' in existing_obj):
            existing_obj_tenant = existing_obj['tenant_ref'].split('#')[1]
            obj_tenant = obj['tenant_ref'].split('name=')[1]
            if obj_tenant != existing_obj_tenant:
                existing_obj = None
    else:
        existing_obj = api.get(obj_path, tenant=tenant, tenant_uuid=tenant_uuid, params={'include_refs': '', 'include_name': ''}, api_version=api_version).json()
    if state == 'absent':
        rsp = None
        changed = False
        err = False
        if not check_mode and existing_obj:
            try:
                if name is not None:
                    rsp = api.delete_by_name(obj_type, name, tenant=tenant, tenant_uuid=tenant_uuid, api_version=api_version)
                else:
                    rsp = api.delete(obj_path, tenant=tenant, tenant_uuid=tenant_uuid, api_version=api_version)
            except ObjectNotFound:
                pass
        if check_mode and existing_obj:
            changed = True
        if rsp:
            if rsp.status_code == 204:
                changed = True
            else:
                err = True
        if not err:
            return ansible_return(module, rsp, changed, existing_obj=existing_obj, api_context=api.get_context())
        elif rsp:
            return module.fail_json(msg=rsp.text)
    rsp = None
    req = None
    if existing_obj:
        if name is not None and obj_type != 'cluster':
            obj_uuid = existing_obj['uuid']
            obj_path = '%s/%s' % (obj_type, obj_uuid)
        if avi_update_method == 'put':
            changed = not avi_obj_cmp(obj, existing_obj, sensitive_fields)
            obj = cleanup_absent_fields(obj)
            if changed:
                req = obj
                if check_mode:
                    rsp = AviCheckModeResponse(obj=existing_obj)
                else:
                    rsp = api.put(obj_path, data=req, tenant=tenant, tenant_uuid=tenant_uuid, api_version=api_version)
            elif check_mode:
                rsp = AviCheckModeResponse(obj=existing_obj)
        elif check_mode:
            rsp = AviCheckModeResponse(obj=existing_obj)
            changed = True
        else:
            obj.pop('name', None)
            patch_data = {avi_patch_op: obj}
            rsp = api.patch(obj_path, data=patch_data, tenant=tenant, tenant_uuid=tenant_uuid, api_version=api_version)
            obj = rsp.json()
            changed = not avi_obj_cmp(obj, existing_obj)
        if changed:
            log.debug('EXISTING OBJ %s', existing_obj)
            log.debug('NEW OBJ %s', obj)
    else:
        changed = True
        req = obj
        if check_mode:
            rsp = AviCheckModeResponse(obj=None)
        else:
            rsp = api.post(obj_type, data=obj, tenant=tenant, tenant_uuid=tenant_uuid, api_version=api_version)
    return ansible_return(module, rsp, changed, req, existing_obj=existing_obj, api_context=api.get_context())