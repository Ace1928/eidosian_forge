import abc
import copy
from ansible.module_utils.six import raise_from
import importlib
import os
from ansible.module_utils.basic import AnsibleModule
def openstack_full_argument_spec(**kwargs):
    spec = dict(cloud=dict(type='raw'), auth_type=dict(), auth=dict(type='dict', no_log=True), region_name=dict(), validate_certs=dict(type='bool', aliases=['verify']), ca_cert=dict(aliases=['cacert']), client_cert=dict(aliases=['cert']), client_key=dict(no_log=True, aliases=['key']), wait=dict(default=True, type='bool'), timeout=dict(default=180, type='int'), api_timeout=dict(type='int'), interface=dict(default='public', choices=['public', 'internal', 'admin'], aliases=['endpoint_type']), sdk_log_path=dict(), sdk_log_level=dict(default='INFO', choices=['INFO', 'DEBUG']))
    kwargs_copy = copy.deepcopy(kwargs)
    for v in kwargs_copy.values():
        for c in CUSTOM_VAR_PARAMS:
            v.pop(c, None)
    spec.update(kwargs_copy)
    return spec