from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def normalize_macro_name(macro_name):
    if ':' in macro_name:
        macro_name = ':'.join([macro_name.split(':')[0].upper(), ':'.join(macro_name.split(':')[1:])])
    else:
        macro_name = macro_name.upper()
    if not macro_name.startswith('{$'):
        macro_name = '{$' + macro_name
    if not macro_name.endswith('}'):
        macro_name = macro_name + '}'
    return macro_name