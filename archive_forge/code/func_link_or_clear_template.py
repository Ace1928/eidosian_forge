from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def link_or_clear_template(self, host_id, template_id_list, tls_connect, tls_accept, tls_psk_identity, tls_psk, tls_issuer, tls_subject, ipmi_authtype, ipmi_privilege, ipmi_username, ipmi_password, discovered_host):
    exist_template_id_list = self.get_host_templates_by_host_id(host_id)
    exist_template_ids = set(exist_template_id_list)
    template_ids = set(template_id_list)
    template_id_list = list(template_ids)
    templates_clear = exist_template_ids.difference(template_ids)
    templates_clear_list = list(templates_clear)
    if discovered_host:
        request_str = {'hostid': host_id, 'templates': template_id_list, 'templates_clear': templates_clear_list}
    else:
        request_str = {'hostid': host_id, 'templates': template_id_list, 'templates_clear': templates_clear_list, 'ipmi_authtype': ipmi_authtype, 'ipmi_privilege': ipmi_privilege, 'ipmi_username': ipmi_username, 'ipmi_password': ipmi_password}
        if tls_connect:
            request_str['tls_connect'] = tls_connect
        if tls_accept:
            request_str['tls_accept'] = tls_accept
        if tls_psk_identity is not None:
            request_str['tls_psk_identity'] = tls_psk_identity
        if tls_psk is not None:
            request_str['tls_psk'] = tls_psk
        if tls_issuer is not None:
            request_str['tls_issuer'] = tls_issuer
        if tls_subject is not None:
            request_str['tls_subject'] = tls_subject
    try:
        if self._module.check_mode:
            self._module.exit_json(changed=True)
        self._zapi.host.update(request_str)
    except Exception as e:
        self._module.fail_json(msg='Failed to link template to host: %s' % e)