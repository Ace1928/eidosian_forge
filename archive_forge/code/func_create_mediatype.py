from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def create_mediatype(self, **kwargs):
    try:
        self._zapi.mediatype.create(kwargs)
    except Exception as e:
        self._module.fail_json(msg="Failed to create mediatype '{name}': {e}".format(name=kwargs['name'], e=e))