from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_proxy_by_proxy_name(self, proxy_name):
    """Get proxy by proxy name
        Args:
            proxy_name: proxy name.
        Returns:
            proxy matching proxy name
        """
    try:
        proxy_list = self._zapi.proxy.get({'output': 'extend', 'selectInterface': 'extend', 'filter': {'host': [proxy_name]}})
        if len(proxy_list) < 1:
            self._module.fail_json(msg='Proxy not found: %s' % proxy_name)
        else:
            return proxy_list[0]
    except Exception as e:
        self._module.fail_json(msg="Failed to get proxy '%s': %s" % (proxy_name, e))