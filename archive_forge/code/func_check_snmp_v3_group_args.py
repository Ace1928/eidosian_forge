from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def check_snmp_v3_group_args(self, **kwargs):
    """ Check snmp v3 group args """
    module = kwargs['module']
    result = dict()
    need_cfg = False
    result['group_info'] = []
    state = module.params['state']
    group_name = module.params['group_name']
    security_level = module.params['security_level']
    acl_number = module.params['acl_number']
    read_view = module.params['read_view']
    write_view = module.params['write_view']
    notify_view = module.params['notify_view']
    community_name = module.params['community_name']
    access_right = module.params['access_right']
    if group_name and security_level:
        if community_name and access_right:
            module.fail_json(msg='Error: Community is used for v1/v2c, group_name is used for v3, do not input at the same time.')
        if len(group_name) > 32 or len(group_name) == 0:
            module.fail_json(msg='Error: The len of group_name %s is out of [1 - 32].' % group_name)
        if acl_number:
            if acl_number.isdigit():
                if int(acl_number) > 2999 or int(acl_number) < 2000:
                    module.fail_json(msg='Error: The value of acl_number %s is out of [2000 - 2999].' % acl_number)
            elif not acl_number[0].isalpha() or len(acl_number) > 32 or len(acl_number) < 1:
                module.fail_json(msg='Error: The len of acl_number %s is out of [1 - 32] or is invalid.' % acl_number)
        if read_view:
            if len(read_view) > 32 or len(read_view) < 1:
                module.fail_json(msg='Error: The len of read_view %s is out of [1 - 32].' % read_view)
        if write_view:
            if len(write_view) > 32 or len(write_view) < 1:
                module.fail_json(msg='Error: The len of write_view %s is out of [1 - 32].' % write_view)
        if notify_view:
            if len(notify_view) > 32 or len(notify_view) < 1:
                module.fail_json(msg='Error: The len of notify_view %s is out of [1 - 32].' % notify_view)
        conf_str = CE_GET_SNMP_V3_GROUP_HEADER
        if acl_number:
            conf_str += '<aclNumber></aclNumber>'
        if read_view:
            conf_str += '<readViewName></readViewName>'
        if write_view:
            conf_str += '<writeViewName></writeViewName>'
        if notify_view:
            conf_str += '<notifyViewName></notifyViewName>'
        conf_str += CE_GET_SNMP_V3_GROUP_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            if state == 'present':
                need_cfg = True
        else:
            xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
            root = ElementTree.fromstring(xml_str)
            group_info = root.findall('snmp/snmpv3Groups/snmpv3Group')
            if group_info:
                for tmp in group_info:
                    tmp_dict = dict()
                    for site in tmp:
                        if site.tag in ['groupName', 'securityLevel', 'readViewName', 'writeViewName', 'notifyViewName', 'aclNumber']:
                            tmp_dict[site.tag] = site.text
                    result['group_info'].append(tmp_dict)
            if result['group_info']:
                group_name_list = list()
                for tmp in result['group_info']:
                    if 'groupName' in tmp.keys():
                        group_name_list.append(tmp['groupName'])
                if group_name not in group_name_list:
                    if state == 'present':
                        need_cfg = True
                    else:
                        need_cfg = False
                else:
                    need_cfg_bool = True
                    for tmp in result['group_info']:
                        if tmp['groupName'] == group_name:
                            cfg_bool_list = list()
                            if security_level:
                                if 'securityLevel' in tmp.keys():
                                    need_cfg_group = False
                                    if tmp['securityLevel'] != security_level:
                                        need_cfg_group = True
                                else:
                                    need_cfg_group = True
                                cfg_bool_list.append(need_cfg_group)
                            if acl_number:
                                if 'aclNumber' in tmp.keys():
                                    need_cfg_acl = False
                                    if tmp['aclNumber'] != acl_number:
                                        need_cfg_acl = True
                                else:
                                    need_cfg_acl = True
                                cfg_bool_list.append(need_cfg_acl)
                            if read_view:
                                if 'readViewName' in tmp.keys():
                                    need_cfg_read = False
                                    if tmp['readViewName'] != read_view:
                                        need_cfg_read = True
                                else:
                                    need_cfg_read = True
                                cfg_bool_list.append(need_cfg_read)
                            if write_view:
                                if 'writeViewName' in tmp.keys():
                                    need_cfg_write = False
                                    if tmp['writeViewName'] != write_view:
                                        need_cfg_write = True
                                else:
                                    need_cfg_write = True
                                cfg_bool_list.append(need_cfg_write)
                            if notify_view:
                                if 'notifyViewName' in tmp.keys():
                                    need_cfg_notify = False
                                    if tmp['notifyViewName'] != notify_view:
                                        need_cfg_notify = True
                                else:
                                    need_cfg_notify = True
                                cfg_bool_list.append(need_cfg_notify)
                            if True not in cfg_bool_list:
                                need_cfg_bool = False
                    if state == 'present':
                        if not need_cfg_bool:
                            need_cfg = False
                        else:
                            need_cfg = True
                    elif not need_cfg_bool:
                        need_cfg = True
                    else:
                        need_cfg = False
    result['need_cfg'] = need_cfg
    return result