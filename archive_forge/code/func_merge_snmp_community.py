from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def merge_snmp_community(self, **kwargs):
    """ Merge snmp community operation """
    module = kwargs['module']
    community_name = module.params['community_name']
    access_right = module.params['access_right']
    acl_number = module.params['acl_number']
    community_mib_view = module.params['community_mib_view']
    conf_str = CE_MERGE_SNMP_COMMUNITY_HEADER % (community_name, access_right)
    if acl_number:
        conf_str += '<aclNumber>%s</aclNumber>' % acl_number
    if community_mib_view:
        conf_str += '<mibViewName>%s</mibViewName>' % community_mib_view
    conf_str += CE_MERGE_SNMP_COMMUNITY_TAIL
    recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        module.fail_json(msg='Error: Merge snmp community failed.')
    community_safe_name = '******'
    cmd = 'snmp-agent community %s %s' % (access_right, community_safe_name)
    if acl_number:
        cmd += ' acl %s' % acl_number
    if community_mib_view:
        cmd += ' mib-view %s' % community_mib_view
    return cmd