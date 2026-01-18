from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def merge_base_rule(self):
    """ Merge base rule operation """
    conf_str = CE_MERGE_ACL_BASE_RULE_HEADER % (self.acl_name, self.rule_name)
    if self.rule_id:
        conf_str += '<aclRuleID>%s</aclRuleID>' % self.rule_id
    if self.rule_action:
        conf_str += '<aclAction>%s</aclAction>' % self.rule_action
    if self.source_ip:
        conf_str += '<aclSourceIp>%s</aclSourceIp>' % self.source_ip
    if self.src_wild:
        conf_str += '<aclSrcWild>%s</aclSrcWild>' % self.src_wild
    if self.frag_type:
        conf_str += '<aclFragType>%s</aclFragType>' % self.frag_type
    if self.vrf_name:
        conf_str += '<vrfName>%s</vrfName>' % self.vrf_name
    if self.time_range:
        conf_str += '<aclTimeName>%s</aclTimeName>' % self.time_range
    if self.rule_description:
        conf_str += '<aclRuleDescription>%s</aclRuleDescription>' % self.rule_description
    conf_str += '<aclLogFlag>%s</aclLogFlag>' % str(self.log_flag).lower()
    conf_str += CE_MERGE_ACL_BASE_RULE_TAIL
    recv_xml = self.netconf_set_config(conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        self.module.fail_json(msg='Error: Merge acl base rule failed.')
    if self.rule_action:
        cmd = 'rule'
        if self.rule_id:
            cmd += ' %s' % self.rule_id
        cmd += ' %s' % self.rule_action
        if self.frag_type == 'fragment':
            cmd += ' fragment-type fragment'
        if self.source_ip and self.src_wild:
            cmd += ' source %s %s' % (self.source_ip, self.src_wild)
        if self.time_range:
            cmd += ' time-range %s' % self.time_range
        if self.vrf_name:
            cmd += ' vpn-instance %s' % self.vrf_name
        if self.log_flag:
            cmd += ' logging'
        self.updates_cmd.append(cmd)
    if self.rule_description:
        cmd = 'rule %s description %s' % (self.rule_id, self.rule_description)
        self.updates_cmd.append(cmd)
    self.changed = True