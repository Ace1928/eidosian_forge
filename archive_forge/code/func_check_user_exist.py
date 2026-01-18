from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible_collections.community.zabbix.plugins.module_utils.helpers import (
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def check_user_exist(self, username):
    zbx_user = self._zapi.user.get({'output': 'extend', 'filter': {'username': username}, 'getAccess': True, 'selectMedias': 'extend', 'selectUsrgrps': 'extend'})
    return zbx_user