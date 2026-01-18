from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def merge_local_user_info(self, **kwargs):
    """ Merge local user information by netconf """
    module = kwargs['module']
    local_user_name = module.params['local_user_name']
    local_password = module.params['local_password']
    local_service_type = module.params['local_service_type']
    local_ftp_dir = module.params['local_ftp_dir']
    local_user_level = module.params['local_user_level']
    local_user_group = module.params['local_user_group']
    state = module.params['state']
    cmds = []
    conf_str = CE_MERGE_LOCAL_USER_INFO_HEADER % local_user_name
    if local_password:
        conf_str += '<password>%s</password>' % local_password
    if state == 'present':
        cmd = 'local-user %s password cipher %s' % (local_user_name, local_password)
        cmds.append(cmd)
    if local_service_type:
        if local_service_type == 'none':
            conf_str += '<serviceTerminal>false</serviceTerminal>'
            conf_str += '<serviceTelnet>false</serviceTelnet>'
            conf_str += '<serviceFtp>false</serviceFtp>'
            conf_str += '<serviceSsh>false</serviceSsh>'
            conf_str += '<serviceSnmp>false</serviceSnmp>'
            conf_str += '<serviceDot1x>false</serviceDot1x>'
            cmd = 'local-user %s service-type none' % local_user_name
            cmds.append(cmd)
        elif local_service_type == 'dot1x':
            if state == 'present':
                conf_str += '<serviceDot1x>true</serviceDot1x>'
                cmd = 'local-user %s service-type dot1x' % local_user_name
            else:
                conf_str += '<serviceDot1x>false</serviceDot1x>'
                cmd = 'undo local-user %s service-type' % local_user_name
            cmds.append(cmd)
        else:
            option = local_service_type.split(' ')
            for tmp in option:
                if tmp == 'dot1x':
                    module.fail_json(msg='Error: Do not input dot1x with other service type.')
                if tmp == 'none':
                    module.fail_json(msg='Error: Do not input none with other service type.')
                if state == 'present':
                    if tmp == 'ftp':
                        conf_str += '<serviceFtp>true</serviceFtp>'
                        cmd = 'local-user %s service-type ftp' % local_user_name
                    elif tmp == 'snmp':
                        conf_str += '<serviceSnmp>true</serviceSnmp>'
                        cmd = 'local-user %s service-type snmp' % local_user_name
                    elif tmp == 'ssh':
                        conf_str += '<serviceSsh>true</serviceSsh>'
                        cmd = 'local-user %s service-type ssh' % local_user_name
                    elif tmp == 'telnet':
                        conf_str += '<serviceTelnet>true</serviceTelnet>'
                        cmd = 'local-user %s service-type telnet' % local_user_name
                    elif tmp == 'terminal':
                        conf_str += '<serviceTerminal>true</serviceTerminal>'
                        cmd = 'local-user %s service-type terminal' % local_user_name
                    cmds.append(cmd)
                elif tmp == 'ftp':
                    conf_str += '<serviceFtp>false</serviceFtp>'
                elif tmp == 'snmp':
                    conf_str += '<serviceSnmp>false</serviceSnmp>'
                elif tmp == 'ssh':
                    conf_str += '<serviceSsh>false</serviceSsh>'
                elif tmp == 'telnet':
                    conf_str += '<serviceTelnet>false</serviceTelnet>'
                elif tmp == 'terminal':
                    conf_str += '<serviceTerminal>false</serviceTerminal>'
            if state == 'absent':
                cmd = 'undo local-user %s service-type' % local_user_name
                cmds.append(cmd)
    if local_ftp_dir:
        if state == 'present':
            conf_str += '<ftpDir>%s</ftpDir>' % local_ftp_dir
            cmd = 'local-user %s ftp-directory %s' % (local_user_name, local_ftp_dir)
            cmds.append(cmd)
        else:
            conf_str += '<ftpDir></ftpDir>'
            cmd = 'undo local-user %s ftp-directory' % local_user_name
            cmds.append(cmd)
    if local_user_level:
        if state == 'present':
            conf_str += '<userLevel>%s</userLevel>' % local_user_level
            cmd = 'local-user %s level %s' % (local_user_name, local_user_level)
            cmds.append(cmd)
        else:
            conf_str += '<userLevel></userLevel>'
            cmd = 'undo local-user %s level' % local_user_name
            cmds.append(cmd)
    if local_user_group:
        if state == 'present':
            conf_str += '<userGroupName>%s</userGroupName>' % local_user_group
            cmd = 'local-user %s user-group %s' % (local_user_name, local_user_group)
            cmds.append(cmd)
        else:
            conf_str += '<userGroupName></userGroupName>'
            cmd = 'undo local-user %s user-group' % local_user_name
            cmds.append(cmd)
    conf_str += CE_MERGE_LOCAL_USER_INFO_TAIL
    recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        module.fail_json(msg='Error: Merge local user info failed.')
    return cmds