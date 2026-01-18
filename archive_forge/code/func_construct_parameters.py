from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def construct_parameters(self):
    """Translates data to a format suitable for Zabbix API and filters
        the ones that are related to the specified mediatype type.

        Returns:
            A dictionary of arguments that are related to transport type,
            and are in a format that is understandable by Zabbix API.
        """
    truths = {'False': '0', 'True': '1'}
    parameters = dict(status='0' if self._module.params['status'] == 'enabled' else '1', type={'email': '0', 'script': '1', 'sms': '2', 'jabber': '3', 'webhook': '4', 'ez_texting': '100'}.get(self._module.params['type']))
    parameters.update(dict(name=self._module.params['name'], description=self._module.params['description'], maxsessions=str(self._module.params['max_sessions']), maxattempts=str(self._module.params['max_attempts']), attempt_interval=str(self._module.params['attempt_interval'])))
    if self._module.params['message_templates']:
        msg_templates = []
        for template in self._module.params['message_templates']:
            msg_templates.append(dict(eventsource={'triggers': '0', 'discovery': '1', 'autoregistration': '2', 'internal': '3'}.get(template['eventsource']), recovery={'operations': '0', 'recovery_operations': '1', 'update_operations': '2'}.get(template['recovery']), subject=template['subject'], message=template['body']))
        parameters.update(dict(message_templates=msg_templates))
    if self._module.params['type'] == 'email':
        parameters.update(dict(smtp_server=self._module.params['smtp_server'], smtp_port=str(self._module.params['smtp_server_port']), smtp_helo=self._module.params['smtp_helo'], smtp_email=self._module.params['smtp_email'], smtp_security={'None': '0', 'STARTTLS': '1', 'SSL/TLS': '2'}.get(str(self._module.params['smtp_security'])), smtp_authentication=truths.get(str(self._module.params['smtp_authentication'])), smtp_verify_host=truths.get(str(self._module.params['smtp_verify_host'])), smtp_verify_peer=truths.get(str(self._module.params['smtp_verify_peer'])), username=self._module.params['username'], passwd=self._module.params['password']))
        if parameters['smtp_authentication'] == '0':
            parameters.pop('username')
            parameters.pop('passwd')
        return parameters
    elif self._module.params['type'] == 'script':
        if LooseVersion(self._zbx_api_version) < LooseVersion('6.4'):
            if self._module.params['script_params'] is None:
                _script_params = ''
            else:
                _script_params = '\n'.join((str(i) for i in self._module.params['script_params'])) + '\n'
            parameters.update(dict(exec_path=self._module.params['script_name'], exec_params=_script_params))
        else:
            _script_params = []
            if self._module.params['script_params']:
                for i, val in enumerate(self._module.params['script_params']):
                    _script_params.append({'sortorder': str(i), 'value': val})
            parameters.update(dict(exec_path=self._module.params['script_name'], parameters=_script_params))
        return parameters
    elif self._module.params['type'] == 'sms':
        parameters.update(dict(gsm_modem=self._module.params['gsm_modem']))
        return parameters
    elif self._module.params['type'] == 'webhook':
        parameters.update(dict(script=self._module.params['webhook_script'], timeout=self._module.params['webhook_timeout'], process_tags=truths.get(str(self._module.params['process_tags'])), show_event_menu=truths.get(str(self._module.params['event_menu'])), parameters=self._module.params['webhook_params']))
        if self._module.params['event_menu']:
            parameters.update(dict(event_menu_url=self._module.params['event_menu_url'], event_menu_name=self._module.params['event_menu_name']))
        return parameters
    self._module.fail_json(msg='%s is unsupported for Zabbix version %s' % (self._module.params['type'], self._zbx_api_version))