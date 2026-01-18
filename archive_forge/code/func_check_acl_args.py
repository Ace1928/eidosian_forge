from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def check_acl_args(self):
    """ Check acl invalid args """
    need_cfg = False
    find_flag = False
    self.cur_acl_cfg['acl_info'] = []
    if self.acl_name:
        if self.acl_name.isdigit():
            if int(self.acl_name) < 3000 or int(self.acl_name) > 3999:
                self.module.fail_json(msg='Error: The value of acl_name is out of [3000-3999] for advance ACL.')
            if self.acl_num:
                self.module.fail_json(msg='Error: The acl_name is digit, so should not input acl_num at the same time.')
        else:
            self.acl_type = 'Advance'
            if len(self.acl_name) < 1 or len(self.acl_name) > 32:
                self.module.fail_json(msg='Error: The len of acl_name is out of [1 - 32].')
            if self.state == 'present':
                if not self.acl_num and (not self.acl_type) and (not self.rule_name):
                    self.module.fail_json(msg='Error: Please input acl_num or acl_type when config ACL.')
        if self.acl_num:
            if self.acl_num.isdigit():
                if int(self.acl_num) < 3000 or int(self.acl_num) > 3999:
                    self.module.fail_json(msg='Error: The value of acl_name is out of [3000-3999] for advance ACL.')
            else:
                self.module.fail_json(msg='Error: The acl_num is not digit.')
        if self.acl_step:
            if self.acl_step.isdigit():
                if int(self.acl_step) < 1 or int(self.acl_step) > 20:
                    self.module.fail_json(msg='Error: The value of acl_step is out of [1 - 20].')
            else:
                self.module.fail_json(msg='Error: The acl_step is not digit.')
        if self.acl_description:
            if len(self.acl_description) < 1 or len(self.acl_description) > 127:
                self.module.fail_json(msg='Error: The len of acl_description is out of [1 - 127].')
        conf_str = CE_GET_ACL_HEADER
        if self.acl_type:
            conf_str += '<aclType></aclType>'
        if self.acl_num or self.acl_name.isdigit():
            conf_str += '<aclNumber></aclNumber>'
        if self.acl_step:
            conf_str += '<aclStep></aclStep>'
        if self.acl_description:
            conf_str += '<aclDescription></aclDescription>'
        conf_str += CE_GET_ACL_TAIL
        recv_xml = self.netconf_get_config(conf_str=conf_str)
        if '<data/>' in recv_xml:
            find_flag = False
        else:
            xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
            root = ElementTree.fromstring(xml_str)
            acl_info = root.findall('acl/aclGroups/aclGroup')
            if acl_info:
                for tmp in acl_info:
                    tmp_dict = dict()
                    for site in tmp:
                        if site.tag in ['aclNumOrName', 'aclType', 'aclNumber', 'aclStep', 'aclDescription']:
                            tmp_dict[site.tag] = site.text
                    self.cur_acl_cfg['acl_info'].append(tmp_dict)
            if self.cur_acl_cfg['acl_info']:
                find_list = list()
                for tmp in self.cur_acl_cfg['acl_info']:
                    cur_cfg_dict = dict()
                    exist_cfg_dict = dict()
                    if self.acl_name:
                        if self.acl_name.isdigit() and tmp.get('aclNumber'):
                            cur_cfg_dict['aclNumber'] = self.acl_name
                            exist_cfg_dict['aclNumber'] = tmp.get('aclNumber')
                        else:
                            cur_cfg_dict['aclNumOrName'] = self.acl_name
                            exist_cfg_dict['aclNumOrName'] = tmp.get('aclNumOrName')
                    if self.acl_type:
                        cur_cfg_dict['aclType'] = self.acl_type
                        exist_cfg_dict['aclType'] = tmp.get('aclType')
                    if self.acl_num:
                        cur_cfg_dict['aclNumber'] = self.acl_num
                        exist_cfg_dict['aclNumber'] = tmp.get('aclNumber')
                    if self.acl_step:
                        cur_cfg_dict['aclStep'] = self.acl_step
                        exist_cfg_dict['aclStep'] = tmp.get('aclStep')
                    if self.acl_description:
                        cur_cfg_dict['aclDescription'] = self.acl_description
                        exist_cfg_dict['aclDescription'] = tmp.get('aclDescription')
                    if cur_cfg_dict == exist_cfg_dict:
                        find_bool = True
                    else:
                        find_bool = False
                    find_list.append(find_bool)
                for mem in find_list:
                    if mem:
                        find_flag = True
                        break
                    else:
                        find_flag = False
            else:
                find_flag = False
    if self.state == 'present':
        need_cfg = bool(not find_flag)
    elif self.state == 'delete_acl':
        need_cfg = bool(find_flag)
    else:
        need_cfg = False
    self.cur_acl_cfg['need_cfg'] = need_cfg