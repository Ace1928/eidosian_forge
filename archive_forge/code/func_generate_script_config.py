from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def generate_script_config(self, name, script_type, command, scope, execute_on, menu_path, authtype, username, password, publickey, privatekey, port, host_group, user_group, host_access, confirmation, script_timeout, parameters, description):
    if host_group == 'all':
        groupid = '0'
    else:
        groups = self._zapi.hostgroup.get({'filter': {'name': host_group}})
        if not groups:
            self._module.fail_json(changed=False, msg="Host group '%s' not found" % host_group)
        groupid = groups[0]['groupid']
    if user_group == 'all':
        usrgrpid = '0'
    else:
        user_groups = self._zapi.usergroup.get({'filter': {'name': user_group}})
        if not user_groups:
            self._module.fail_json(changed=False, msg="User group '%s' not found" % user_group)
        usrgrpid = user_groups[0]['usrgrpid']
    request = {'name': name, 'type': str(zabbix_utils.helper_to_numeric_value(['script', 'ipmi', 'ssh', 'telnet', '', 'webhook'], script_type)), 'command': command, 'scope': str(zabbix_utils.helper_to_numeric_value(['', 'action_operation', 'manual_host_action', '', 'manual_event_action'], scope)), 'groupid': groupid}
    if description is not None:
        request['description'] = description
    if script_type == 'script':
        if execute_on is None:
            execute_on = 'zabbix_server_proxy'
        request['execute_on'] = str(zabbix_utils.helper_to_numeric_value(['zabbix_agent', 'zabbix_server', 'zabbix_server_proxy'], execute_on))
    if scope in ['manual_host_action', 'manual_event_action']:
        if menu_path is None:
            request['menu_path'] = ''
        else:
            request['menu_path'] = menu_path
        request['usrgrpid'] = usrgrpid
        request['host_access'] = str(zabbix_utils.helper_to_numeric_value(['', '', 'read', 'write'], host_access))
        if confirmation is None:
            request['confirmation'] = ''
        else:
            request['confirmation'] = confirmation
    if script_type == 'ssh':
        if authtype is None:
            self._module.fail_json(changed=False, msg='authtype must be provided for ssh script type')
        request['authtype'] = str(zabbix_utils.helper_to_numeric_value(['password', 'public_key'], authtype))
        if authtype == 'public_key':
            if publickey is None or privatekey is None:
                self._module.fail_json(changed=False, msg='publickey and privatekey must be provided for ssh script type with publickey authtype')
            request['publickey'] = publickey
            request['privatekey'] = privatekey
    if script_type in ['ssh', 'telnet']:
        if username is None:
            self._module.fail_json(changed=False, msg="username must be provided for 'ssh' and 'telnet' script types")
        request['username'] = username
        if script_type == 'ssh' and authtype == 'password' or script_type == 'telnet':
            if password is None:
                self._module.fail_json(changed=False, msg='password must be provided for telnet script type or ssh script type with password autheype')
            request['password'] = password
        if port is not None:
            request['port'] = port
    if script_type == 'webhook':
        request['timeout'] = script_timeout
        if parameters:
            request['parameters'] = parameters
    return request