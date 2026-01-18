from __future__ import absolute_import, division, print_function
import json
import traceback
import re
import xml.etree.ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.six import PY2
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_group_ids_by_group_names(self, group_names):
    group_ids = []
    for group_name in group_names:
        if LooseVersion(self._zbx_api_version) >= LooseVersion('6.2'):
            group = self._zapi.templategroup.get({'output': ['groupid'], 'filter': {'name': group_name}})
        else:
            group = self._zapi.hostgroup.get({'output': ['groupid'], 'filter': {'name': group_name}})
        if group:
            group_ids.append({'groupid': group[0]['groupid']})
        else:
            self._module.fail_json(msg='Template group not found: %s' % group_name)
    return group_ids