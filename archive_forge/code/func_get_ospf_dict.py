from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_ospf_dict(self, process_id):
    """ get one ospf attributes dict."""
    ospf_info = dict()
    conf_str = CE_NC_GET_OSPF % process_id
    xml_str = get_nc_config(self.module, conf_str)
    if '<data/>' in xml_str:
        return ospf_info
    xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    ospfsite = root.find('ospfv2/ospfv2comm/ospfSites/ospfSite')
    if ospfsite:
        for site in ospfsite:
            if site.tag in ['processId', 'routerId', 'vrfName']:
                ospf_info[site.tag] = site.text
    topo = root.find('ospfv2/ospfv2comm/ospfSites/ospfSite/ProcessTopologys/ProcessTopology')
    if topo:
        for eles in topo:
            if eles.tag in ['maxLoadBalancing']:
                ospf_info[eles.tag] = eles.text
    ospf_info['nexthops'] = list()
    nexthops = root.findall('ospfv2/ospfv2comm/ospfSites/ospfSite/ProcessTopologys/ProcessTopology/nexthopMTs/nexthopMT')
    if nexthops:
        for nexthop in nexthops:
            nh_dict = dict()
            for ele in nexthop:
                if ele.tag in ['ipAddress', 'weight']:
                    nh_dict[ele.tag] = ele.text
            ospf_info['nexthops'].append(nh_dict)
    ospf_info['areas'] = list()
    areas = root.findall('ospfv2/ospfv2comm/ospfSites/ospfSite/areas/area')
    if areas:
        for area in areas:
            area_dict = dict()
            for ele in area:
                if ele.tag in ['areaId', 'authTextSimple', 'areaType', 'authenticationMode', 'keyId', 'authTextMd5']:
                    area_dict[ele.tag] = ele.text
                if ele.tag == 'networks':
                    area_dict['networks'] = list()
                    for net in ele:
                        net_dict = dict()
                        for net_ele in net:
                            if net_ele.tag in ['ipAddress', 'wildcardMask']:
                                net_dict[net_ele.tag] = net_ele.text
                        area_dict['networks'].append(net_dict)
            ospf_info['areas'].append(area_dict)
    return ospf_info