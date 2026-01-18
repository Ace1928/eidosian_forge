from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def check_snmp_v3_usm_user_args(self, **kwargs):
    """ Check snmp v3 usm user invalid args """
    module = kwargs['module']
    result = dict()
    need_cfg = False
    state = module.params['state']
    usm_user_name = module.params['usm_user_name']
    remote_engine_id = module.params['remote_engine_id']
    acl_number = module.params['acl_number']
    user_group = module.params['user_group']
    auth_protocol = module.params['auth_protocol']
    auth_key = module.params['auth_key']
    priv_protocol = module.params['priv_protocol']
    priv_key = module.params['priv_key']
    local_user_name = module.params['aaa_local_user']
    if usm_user_name:
        if len(usm_user_name) > 32 or len(usm_user_name) == 0:
            module.fail_json(msg='Error: The length of usm_user_name %s is out of [1 - 32].' % usm_user_name)
        if remote_engine_id:
            if len(remote_engine_id) > 64 or len(remote_engine_id) < 10:
                module.fail_json(msg='Error: The length of remote_engine_id %s is out of [10 - 64].' % remote_engine_id)
        conf_str = CE_GET_SNMP_V3_USM_USER_HEADER
        if acl_number:
            if acl_number.isdigit():
                if int(acl_number) > 2999 or int(acl_number) < 2000:
                    module.fail_json(msg='Error: The value of acl_number %s is out of [2000 - 2999].' % acl_number)
            elif not acl_number[0].isalpha() or len(acl_number) > 32 or len(acl_number) < 1:
                module.fail_json(msg='Error: The length of acl_number %s is out of [1 - 32].' % acl_number)
            conf_str += '<aclNumber></aclNumber>'
        if user_group:
            if len(user_group) > 32 or len(user_group) == 0:
                module.fail_json(msg='Error: The length of user_group %s is out of [1 - 32].' % user_group)
            conf_str += '<groupName></groupName>'
        if auth_protocol:
            conf_str += '<authProtocol></authProtocol>'
        if auth_key:
            if len(auth_key) > 255 or len(auth_key) == 0:
                module.fail_json(msg='Error: The length of auth_key %s is out of [1 - 255].' % auth_key)
            conf_str += '<authKey></authKey>'
        if priv_protocol:
            if not auth_protocol:
                module.fail_json(msg='Error: Please input auth_protocol at the same time.')
            conf_str += '<privProtocol></privProtocol>'
        if priv_key:
            if len(priv_key) > 255 or len(priv_key) == 0:
                module.fail_json(msg='Error: The length of priv_key %s is out of [1 - 255].' % priv_key)
            conf_str += '<privKey></privKey>'
        result['usm_user_info'] = []
        conf_str += CE_GET_SNMP_V3_USM_USER_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            if state == 'present':
                need_cfg = True
        else:
            xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
            root = ElementTree.fromstring(xml_str)
            usm_user_info = root.findall('snmp/usmUsers/usmUser')
            if usm_user_info:
                for tmp in usm_user_info:
                    tmp_dict = dict()
                    tmp_dict['remoteEngineID'] = None
                    for site in tmp:
                        if site.tag in ['userName', 'remoteEngineID', 'engineID', 'groupName', 'authProtocol', 'authKey', 'privProtocol', 'privKey', 'aclNumber']:
                            tmp_dict[site.tag] = site.text
                    result['usm_user_info'].append(tmp_dict)
            cur_cfg = dict()
            if usm_user_name:
                cur_cfg['userName'] = usm_user_name
            if user_group:
                cur_cfg['groupName'] = user_group
            if auth_protocol:
                cur_cfg['authProtocol'] = auth_protocol
            if auth_key:
                cur_cfg['authKey'] = auth_key
            if priv_protocol:
                cur_cfg['privProtocol'] = priv_protocol
            if priv_key:
                cur_cfg['privKey'] = priv_key
            if acl_number:
                cur_cfg['aclNumber'] = acl_number
            if remote_engine_id:
                cur_cfg['engineID'] = remote_engine_id
                cur_cfg['remoteEngineID'] = 'true'
            else:
                cur_cfg['engineID'] = self.local_engine_id
                cur_cfg['remoteEngineID'] = 'false'
            if result['usm_user_info']:
                num = 0
                for tmp in result['usm_user_info']:
                    if cur_cfg == tmp:
                        num += 1
                if num == 0:
                    if state == 'present':
                        need_cfg = True
                    else:
                        need_cfg = False
                elif state == 'present':
                    need_cfg = False
                else:
                    need_cfg = True
            elif state == 'present':
                need_cfg = True
            else:
                need_cfg = False
    result['need_cfg'] = need_cfg
    return result