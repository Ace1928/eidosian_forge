from __future__ import absolute_import, division, print_function
from ansible.module_utils import distro
from ansible.module_utils.basic import AnsibleModule
def get_xenapi_session():
    session = XenAPI.xapi_local()
    session.xenapi.login_with_password('', '')
    return session