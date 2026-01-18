from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
def fortios_firewall_service(data, fos, check_mode):
    fos.do_member_operation('firewall.service', 'group')
    if data['firewall_service_group']:
        resp = firewall_service_group(data, fos, check_mode)
    else:
        fos._module.fail_json(msg='missing task body: %s' % 'firewall_service_group')
    if isinstance(resp, tuple) and len(resp) == 4:
        return resp
    return (not is_successful_status(resp), is_successful_status(resp) and (resp['revision_changed'] if 'revision_changed' in resp else True), resp, {})