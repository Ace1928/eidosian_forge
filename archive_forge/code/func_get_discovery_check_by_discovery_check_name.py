from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_discovery_check_by_discovery_check_name(self, discovery_check_name):
    """Get discovery check  by discovery check name

        Args:
            discovery_check_name: discovery check name.

        Returns:
            discovery check matching discovery check name

        """
    try:
        discovery_rule_name, dcheck_type = discovery_check_name.split(': ')
        dcheck_type_to_number = {'SSH': '0', 'LDAP': '1', 'SMTP': '2', 'FTP': '3', 'HTTP': '4', 'POP': '5', 'NNTP': '6', 'IMAP': '7', 'TCP': '8', 'Zabbix agent': '9', 'SNMPv1 agent': '10', 'SNMPv2 agent': '11', 'ICMP ping': '12', 'SNMPv3 agent': '13', 'HTTPS': '14', 'Telnet': '15'}
        if dcheck_type not in dcheck_type_to_number:
            self._module.fail_json(msg='Discovery check type: %s does not exist' % dcheck_type)
        discovery_rule_list = self._zapi.drule.get({'output': ['dchecks'], 'filter': {'name': [discovery_rule_name]}, 'selectDChecks': 'extend'})
        if len(discovery_rule_list) < 1:
            self._module.fail_json(msg='Discovery check not found: %s' % discovery_check_name)
        for dcheck in discovery_rule_list[0]['dchecks']:
            if dcheck_type_to_number[dcheck_type] == dcheck['type']:
                return dcheck
        self._module.fail_json(msg='Discovery check not found: %s' % discovery_check_name)
    except Exception as e:
        self._module.fail_json(msg="Failed to get discovery check '%s': %s" % (discovery_check_name, e))