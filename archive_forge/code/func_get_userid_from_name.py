from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_userid_from_name(self, username):
    try:
        userids = self._zapi.user.get({'output': 'userid', 'filter': {'username': username}})
        if not userids or len(userids) > 1:
            self._module.fail_json("User '%s' cannot be found" % username)
        return userids[0]['userid']
    except Exception as e:
        self._module.fail_json(msg='Failed to get userid: %s' % e)