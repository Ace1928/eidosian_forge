from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible_collections.community.zabbix.plugins.module_utils.helpers import (
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_usergroups_by_name(self, usrgrps):
    params = {'output': ['usrgrpid', 'name', 'gui_access'], 'filter': {'name': usrgrps}}
    res = self._zapi.usergroup.get(params)
    if res:
        ids = [{'usrgrpid': g['usrgrpid']} for g in res]
        if bool([g for g in res if g['gui_access'] == '1']):
            require_password = True
        elif bool([g for g in res if g['gui_access'] == '2' or g['gui_access'] == '3']):
            require_password = False
        elif bool([g for g in res if g['gui_access'] == '0']):
            default_authentication = self.get_default_authentication()
            require_password = True if default_authentication == 'internal' else False
        not_found_groups = set(usrgrps) - set([g['name'] for g in res])
        if not_found_groups:
            self._module.fail_json(msg='User groups not found: %s' % not_found_groups)
        return (ids, require_password)
    else:
        self._module.fail_json(msg='No user groups found')