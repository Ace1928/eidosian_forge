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
def fortios_switch_controller_security_policy(data, fos, check_mode):
    fos.do_member_operation('switch-controller.security-policy', 'captive-portal')
    if data['switch_controller_security_policy_captive_portal']:
        resp = switch_controller_security_policy_captive_portal(data, fos, check_mode)
    else:
        fos._module.fail_json(msg='missing task body: %s' % 'switch_controller_security_policy_captive_portal')
    if isinstance(resp, tuple) and len(resp) == 4:
        return resp
    return (not is_successful_status(resp), is_successful_status(resp) and (resp['revision_changed'] if 'revision_changed' in resp else True), resp, {})