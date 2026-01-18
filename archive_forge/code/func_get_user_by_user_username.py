from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_user_by_user_username(self, username):
    zabbix_user = ''
    try:
        data = {'output': 'extend', 'filter': {}, 'getAccess': True, 'selectMedias': 'extend', 'selectUsrgrps': 'extend'}
        data['filter']['username'] = username
        zabbix_user = self._zapi.user.get(data)
    except Exception as e:
        self._zapi.logout()
        self._module.fail_json(msg='Failed to get user information: %s' % e)
    if not zabbix_user:
        zabbix_user = {}
    else:
        zabbix_user = zabbix_user[0]
    return zabbix_user