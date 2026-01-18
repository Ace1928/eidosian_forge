from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def check_snmp_v3_local_user_args(self, **kwargs):
    """ Check snmp v3 local user invalid args """
    module = kwargs['module']
    result = dict()
    need_cfg = False
    state = module.params['state']
    local_user_name = module.params['aaa_local_user']
    auth_protocol = module.params['auth_protocol']
    auth_key = module.params['auth_key']
    priv_protocol = module.params['priv_protocol']
    priv_key = module.params['priv_key']
    usm_user_name = module.params['usm_user_name']
    if local_user_name:
        if usm_user_name:
            module.fail_json(msg='Error: Please do not input usm_user_name and local_user_name at the same time.')
        if not auth_protocol or not auth_key or (not priv_protocol) or (not priv_key):
            module.fail_json(msg='Error: Please input auth_protocol auth_key priv_protocol priv_key for local user.')
        if len(local_user_name) > 32 or len(local_user_name) == 0:
            module.fail_json(msg='Error: The length of local_user_name %s is out of [1 - 32].' % local_user_name)
        if len(auth_key) > 255 or len(auth_key) == 0:
            module.fail_json(msg='Error: The length of auth_key %s is out of [1 - 255].' % auth_key)
        if len(priv_key) > 255 or len(priv_key) == 0:
            module.fail_json(msg='Error: The length of priv_key %s is out of [1 - 255].' % priv_key)
        result['local_user_info'] = []
        conf_str = CE_GET_SNMP_V3_LOCAL_USER
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            if state == 'present':
                need_cfg = True
        else:
            xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
            root = ElementTree.fromstring(xml_str)
            local_user_info = root.findall('snmp/localUsers/localUser')
            if local_user_info:
                for tmp in local_user_info:
                    tmp_dict = dict()
                    for site in tmp:
                        if site.tag in ['userName', 'authProtocol', 'authKey', 'privProtocol', 'privKey']:
                            tmp_dict[site.tag] = site.text
                    result['local_user_info'].append(tmp_dict)
            if result['local_user_info']:
                for tmp in result['local_user_info']:
                    if 'userName' in tmp.keys():
                        if state == 'present':
                            if tmp['userName'] != local_user_name:
                                need_cfg = True
                        elif tmp['userName'] == local_user_name:
                            need_cfg = True
                    if auth_protocol:
                        if 'authProtocol' in tmp.keys():
                            if state == 'present':
                                if tmp['authProtocol'] != auth_protocol:
                                    need_cfg = True
                            elif tmp['authProtocol'] == auth_protocol:
                                need_cfg = True
                    if auth_key:
                        if 'authKey' in tmp.keys():
                            if state == 'present':
                                if tmp['authKey'] != auth_key:
                                    need_cfg = True
                            elif tmp['authKey'] == auth_key:
                                need_cfg = True
                    if priv_protocol:
                        if 'privProtocol' in tmp.keys():
                            if state == 'present':
                                if tmp['privProtocol'] != priv_protocol:
                                    need_cfg = True
                            elif tmp['privProtocol'] == priv_protocol:
                                need_cfg = True
                    if priv_key:
                        if 'privKey' in tmp.keys():
                            if state == 'present':
                                if tmp['privKey'] != priv_key:
                                    need_cfg = True
                            elif tmp['privKey'] == priv_key:
                                need_cfg = True
    result['need_cfg'] = need_cfg
    return result