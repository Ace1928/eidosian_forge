from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def get_snmp_local_engine(self, **kwargs):
    """ Get snmp local engine operation """
    module = kwargs['module']
    conf_str = GET_SNMP_LOCAL_ENGINE
    recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
    if '</data>' in recv_xml:
        xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        local_engine_info = root.findall('snmp/engine/engineID')
        if local_engine_info:
            self.local_engine_id = local_engine_info[0].text