from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def set_vlanview_igmp(self):
    """set igmp of vlanview"""
    if not self.changed:
        return
    addr_family = self.addr_family
    state = self.state
    igmp_xml = '\n'
    version_xml = '\n'
    proxy_xml = '\n'
    if state == 'present':
        if self.igmp:
            igmp_xml = get_xml(CE_NC_MERGE_IGMP_VLANVIEW_SNOENABLE, self.igmp.lower())
        if str(self.version):
            version_xml = get_xml(CE_NC_MERGE_IGMP_VLANVIEW_VERSION, self.version)
        if self.proxy:
            proxy_xml = get_xml(CE_NC_MERGE_IGMP_VLANVIEW_PROXYENABLE, self.proxy.lower())
        configxmlstr = CE_NC_MERGE_IGMP_VLANVIEW % (addr_family, self.vlan_id, igmp_xml, version_xml, proxy_xml)
    else:
        configxmlstr = CE_NC_DELETE_IGMP_VLANVIEW % (addr_family, self.vlan_id)
    conf_str = build_config_xml(configxmlstr)
    recv_xml = set_nc_config(self.module, conf_str)
    self._checkresponse_(recv_xml, 'SET_VLANVIEW_IGMP')