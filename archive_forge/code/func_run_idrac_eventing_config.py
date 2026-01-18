from __future__ import (absolute_import, division, print_function)
import os
import tempfile
import json
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def run_idrac_eventing_config(idrac, module):
    """
    Get Lifecycle Controller status

    Keyword arguments:
    idrac  -- iDRAC handle
    module -- Ansible module
    """
    idrac.use_redfish = True
    share_path = tempfile.gettempdir() + os.sep
    upd_share = file_share_manager.create_share_obj(share_path=share_path, isFolder=True)
    if not upd_share.IsValid:
        module.fail_json(msg='Unable to access the share. Ensure that the share name, share mount, and share credentials provided are correct.')
    set_liason = idrac.config_mgr.set_liason_share(upd_share)
    if set_liason['Status'] == 'Failed':
        try:
            message = set_liason['Data']['Message']
        except (IndexError, KeyError):
            message = set_liason['Message']
        module.fail_json(msg=message)
    if module.params['destination_number'] is not None:
        if module.params['destination'] is not None:
            idrac.config_mgr.configure_snmp_trap_destination(destination=module.params['destination'], destination_number=module.params['destination_number'])
        if module.params['snmp_v3_username'] is not None:
            idrac.config_mgr.configure_snmp_trap_destination(snmp_v3_username=module.params['snmp_v3_username'], destination_number=module.params['destination_number'])
        if module.params['snmp_trap_state'] is not None:
            idrac.config_mgr.configure_snmp_trap_destination(state=State_SNMPAlertTypes[module.params['snmp_trap_state']], destination_number=module.params['destination_number'])
    if module.params['alert_number'] is not None:
        if module.params['email_alert_state'] is not None:
            idrac.config_mgr.configure_email_alerts(state=Enable_EmailAlertTypes[module.params['email_alert_state']], alert_number=module.params['alert_number'])
        if module.params['address'] is not None:
            idrac.config_mgr.configure_email_alerts(address=module.params['address'], alert_number=module.params['alert_number'])
        if module.params['custom_message'] is not None:
            idrac.config_mgr.configure_email_alerts(custom_message=module.params['custom_message'], alert_number=module.params['alert_number'])
    if module.params['enable_alerts'] is not None:
        idrac.config_mgr.configure_idrac_alerts(enable_alerts=AlertEnable_IPMILanTypes[module.params['enable_alerts']])
    if module.params['authentication'] is not None:
        idrac.config_mgr.configure_smtp_server_settings(authentication=SMTPAuthentication_RemoteHostsTypes[module.params['authentication']])
    if module.params['smtp_ip_address'] is not None:
        idrac.config_mgr.configure_smtp_server_settings(smtp_ip_address=module.params['smtp_ip_address'])
    if module.params['smtp_port'] is not None:
        idrac.config_mgr.configure_smtp_server_settings(smtp_port=module.params['smtp_port'])
    if module.params['username'] is not None:
        idrac.config_mgr.configure_smtp_server_settings(username=module.params['username'])
    if module.params['password'] is not None:
        idrac.config_mgr.configure_smtp_server_settings(password=module.params['password'])
    if module.check_mode:
        status = idrac.config_mgr.is_change_applicable()
        if status.get('changes_applicable'):
            module.exit_json(msg='Changes found to commit!', changed=True)
        else:
            module.exit_json(msg='No changes found to commit!')
    else:
        status = idrac.config_mgr.apply_changes(reboot=False)
    return status