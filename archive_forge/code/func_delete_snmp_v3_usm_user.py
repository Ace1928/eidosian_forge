from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def delete_snmp_v3_usm_user(self, **kwargs):
    """ Delete snmp v3 usm user operation """
    module = kwargs['module']
    usm_user_name = module.params['usm_user_name']
    remote_engine_id = module.params['remote_engine_id']
    acl_number = module.params['acl_number']
    user_group = module.params['user_group']
    auth_protocol = module.params['auth_protocol']
    auth_key = module.params['auth_key']
    priv_protocol = module.params['priv_protocol']
    priv_key = module.params['priv_key']
    if remote_engine_id:
        conf_str = CE_DELETE_SNMP_V3_USM_USER_HEADER % (usm_user_name, 'true', remote_engine_id)
        cmd = 'undo snmp-agent remote-engineid %s usm-user v3 %s' % (remote_engine_id, usm_user_name)
    else:
        if not self.local_engine_id:
            module.fail_json(msg='Error: The local engine id is null, please input remote_engine_id.')
        conf_str = CE_DELETE_SNMP_V3_USM_USER_HEADER % (usm_user_name, 'false', self.local_engine_id)
        cmd = 'undo snmp-agent usm-user v3 %s' % usm_user_name
    if user_group:
        conf_str += '<groupName>%s</groupName>' % user_group
    if acl_number:
        conf_str += '<aclNumber>%s</aclNumber>' % acl_number
    if auth_protocol:
        conf_str += '<authProtocol>%s</authProtocol>' % auth_protocol
    if auth_key:
        conf_str += '<authKey>%s</authKey>' % auth_key
    if priv_protocol:
        conf_str += '<privProtocol>%s</privProtocol>' % priv_protocol
    if priv_key:
        conf_str += '<privKey>%s</privKey>' % priv_key
    conf_str += CE_DELETE_SNMP_V3_USM_USER_TAIL
    recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        module.fail_json(msg='Error: Delete snmp v3 usm user failed.')
    return cmd