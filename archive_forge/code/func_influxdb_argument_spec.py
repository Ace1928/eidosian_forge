from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
@staticmethod
def influxdb_argument_spec():
    return dict(hostname=dict(type='str', default='localhost'), port=dict(type='int', default=8086), path=dict(type='str', default=''), username=dict(type='str', default='root', aliases=['login_username']), password=dict(type='str', default='root', no_log=True, aliases=['login_password']), ssl=dict(type='bool', default=False), validate_certs=dict(type='bool', default=True), timeout=dict(type='int'), retries=dict(type='int', default=3), proxies=dict(type='dict', default={}), use_udp=dict(type='bool', default=False), udp_port=dict(type='int', default=4444))