import abc
import copy
from ansible.module_utils.six import raise_from
import importlib
import os
from ansible.module_utils.basic import AnsibleModule
def openstack_argument_spec():
    OS_AUTH_URL = os.environ.get('OS_AUTH_URL', 'http://127.0.0.1:35357/v2.0/')
    OS_PASSWORD = os.environ.get('OS_PASSWORD', None)
    OS_REGION_NAME = os.environ.get('OS_REGION_NAME', None)
    OS_USERNAME = os.environ.get('OS_USERNAME', 'admin')
    OS_TENANT_NAME = os.environ.get('OS_TENANT_NAME', OS_USERNAME)
    spec = dict(login_username=dict(default=OS_USERNAME), auth_url=dict(default=OS_AUTH_URL), region_name=dict(default=OS_REGION_NAME))
    if OS_PASSWORD:
        spec['login_password'] = dict(default=OS_PASSWORD)
    else:
        spec['login_password'] = dict(required=True)
    if OS_TENANT_NAME:
        spec['login_tenant_name'] = dict(default=OS_TENANT_NAME)
    else:
        spec['login_tenant_name'] = dict(required=True)
    return spec