from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def get_local_user_info(self, **kwargs):
    """ Get local user information """
    module = kwargs['module']
    local_user_name = module.params['local_user_name']
    local_service_type = module.params['local_service_type']
    local_ftp_dir = module.params['local_ftp_dir']
    local_user_level = module.params['local_user_level']
    local_user_group = module.params['local_user_group']
    state = module.params['state']
    result = dict()
    result['local_user_info'] = []
    need_cfg = False
    conf_str = CE_GET_LOCAL_USER_INFO_HEADER
    if local_service_type:
        if local_service_type == 'none':
            conf_str += '<serviceTerminal></serviceTerminal>'
            conf_str += '<serviceTelnet></serviceTelnet>'
            conf_str += '<serviceFtp></serviceFtp>'
            conf_str += '<serviceSsh></serviceSsh>'
            conf_str += '<serviceSnmp></serviceSnmp>'
            conf_str += '<serviceDot1x></serviceDot1x>'
        elif local_service_type == 'dot1x':
            conf_str += '<serviceDot1x></serviceDot1x>'
        else:
            option = local_service_type.split(' ')
            for tmp in option:
                if tmp == 'dot1x':
                    module.fail_json(msg='Error: Do not input dot1x with other service type.')
                elif tmp == 'none':
                    module.fail_json(msg='Error: Do not input none with other service type.')
                elif tmp == 'ftp':
                    conf_str += '<serviceFtp></serviceFtp>'
                elif tmp == 'snmp':
                    conf_str += '<serviceSnmp></serviceSnmp>'
                elif tmp == 'ssh':
                    conf_str += '<serviceSsh></serviceSsh>'
                elif tmp == 'telnet':
                    conf_str += '<serviceTelnet></serviceTelnet>'
                elif tmp == 'terminal':
                    conf_str += '<serviceTerminal></serviceTerminal>'
                else:
                    module.fail_json(msg='Error: Do not support the type [%s].' % tmp)
    if local_ftp_dir:
        conf_str += '<ftpDir></ftpDir>'
    if local_user_level:
        conf_str += '<userLevel></userLevel>'
    if local_user_group:
        conf_str += '<userGroupName></userGroupName>'
    conf_str += CE_GET_LOCAL_USER_INFO_TAIL
    recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
    if '<data/>' in recv_xml:
        if state == 'present':
            need_cfg = True
    else:
        xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        local_user_info = root.findall('aaa/lam/users/user')
        if local_user_info:
            for tmp in local_user_info:
                tmp_dict = dict()
                for site in tmp:
                    if site.tag in ['userName', 'password', 'userLevel', 'ftpDir', 'userGroupName', 'serviceTerminal', 'serviceTelnet', 'serviceFtp', 'serviceSsh', 'serviceSnmp', 'serviceDot1x']:
                        tmp_dict[site.tag] = site.text
                result['local_user_info'].append(tmp_dict)
        if state == 'present':
            need_cfg = True
        elif result['local_user_info']:
            for tmp in result['local_user_info']:
                if 'userName' in tmp.keys():
                    if tmp['userName'] == local_user_name:
                        if not local_service_type and (not local_user_level) and (not local_ftp_dir) and (not local_user_group):
                            need_cfg = True
                        if local_service_type:
                            if local_service_type == 'none':
                                if tmp.get('serviceTerminal') == 'true' or tmp.get('serviceTelnet') == 'true' or tmp.get('serviceFtp') == 'true' or (tmp.get('serviceSsh') == 'true') or (tmp.get('serviceSnmp') == 'true') or (tmp.get('serviceDot1x') == 'true'):
                                    need_cfg = True
                            elif local_service_type == 'dot1x':
                                if tmp.get('serviceDot1x') == 'true':
                                    need_cfg = True
                            elif tmp == 'ftp':
                                if tmp.get('serviceFtp') == 'true':
                                    need_cfg = True
                            elif tmp == 'snmp':
                                if tmp.get('serviceSnmp') == 'true':
                                    need_cfg = True
                            elif tmp == 'ssh':
                                if tmp.get('serviceSsh') == 'true':
                                    need_cfg = True
                            elif tmp == 'telnet':
                                if tmp.get('serviceTelnet') == 'true':
                                    need_cfg = True
                            elif tmp == 'terminal':
                                if tmp.get('serviceTerminal') == 'true':
                                    need_cfg = True
                        if local_user_level:
                            if tmp.get('userLevel') == local_user_level:
                                need_cfg = True
                        if local_ftp_dir:
                            if tmp.get('ftpDir') == local_ftp_dir:
                                need_cfg = True
                        if local_user_group:
                            if tmp.get('userGroupName') == local_user_group:
                                need_cfg = True
                        break
    result['need_cfg'] = need_cfg
    return result