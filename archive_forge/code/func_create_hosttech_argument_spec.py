from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.dns.plugins.module_utils.argspec import (
from ansible_collections.community.dns.plugins.module_utils.provider import (
from ansible_collections.community.dns.plugins.module_utils.wsdl import (
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
from ansible_collections.community.dns.plugins.module_utils.hosttech.wsdl_api import (
from ansible_collections.community.dns.plugins.module_utils.hosttech.json_api import (
def create_hosttech_argument_spec():
    return ArgumentSpec(argument_spec=dict(hosttech_username=dict(type='str'), hosttech_password=dict(type='str', no_log=True), hosttech_token=dict(type='str', no_log=True, aliases=['api_token'])), required_together=[('hosttech_username', 'hosttech_password')], mutually_exclusive=[('hosttech_username', 'hosttech_token')])