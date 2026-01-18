from __future__ import absolute_import, division, print_function
import re
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
def get_nvmeof_interfaces(self):
    """Retrieve all interfaces that are using nvmeof"""
    ifaces = list()
    try:
        rc, ifaces = self.request('storage-systems/%s/interfaces?channelType=hostside' % self.ssid)
    except Exception as error:
        self.module.fail_json(msg='Failed to retrieve defined host interfaces. Array Id [%s]. Error [%s].' % (self.ssid, to_native(error)))
    nvmeof_ifaces = []
    for iface in ifaces:
        interface_type = iface['ioInterfaceTypeData']['interfaceType']
        properties = iface['commandProtocolPropertiesList']['commandProtocolProperties']
        try:
            link_status = iface['ioInterfaceTypeData']['ib']['linkState']
        except Exception as error:
            link_status = iface['ioInterfaceTypeData']['ethernet']['interfaceData']['ethernetData']['linkStatus']
        if properties and properties[0]['commandProtocol'] == 'nvme' and (properties[0]['nvmeProperties']['commandSet'] == 'nvmeof'):
            nvmeof_ifaces.append({'properties': properties[0]['nvmeProperties']['nvmeofProperties'], 'reference': iface['interfaceRef'], 'channel': iface['ioInterfaceTypeData'][iface['ioInterfaceTypeData']['interfaceType']]['channel'], 'interface_type': interface_type, 'interface': iface['ioInterfaceTypeData'][interface_type], 'controller_id': iface['controllerRef'], 'link_status': link_status})
    return nvmeof_ifaces