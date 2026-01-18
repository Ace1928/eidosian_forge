from __future__ import absolute_import, division, print_function
import re
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
def get_target_interface(self):
    """Search for the selected IB iSER interface"""
    if self.get_target_interface_cache is None:
        ifaces = self.get_interfaces()
        ifaces_status = self.get_ib_link_status()
        controller_id = self.get_controllers()[self.controller]
        controller_ifaces = []
        for iface in ifaces:
            if iface['ioInterfaceTypeData']['interfaceType'] == 'iscsi' and iface['controllerRef'] == controller_id:
                controller_ifaces.append([iface['ioInterfaceTypeData']['iscsi']['channel'], iface, ifaces_status[iface['ioInterfaceTypeData']['iscsi']['channelPortRef']]])
            elif iface['ioInterfaceTypeData']['interfaceType'] == 'ib' and iface['controllerRef'] == controller_id:
                controller_ifaces.append([iface['ioInterfaceTypeData']['ib']['channel'], iface, iface['ioInterfaceTypeData']['ib']['linkState']])
        sorted_controller_ifaces = sorted(controller_ifaces)
        if self.channel < 1 or self.channel > len(controller_ifaces):
            status_msg = ', '.join(['%s (link %s)' % (index + 1, values[2]) for index, values in enumerate(sorted_controller_ifaces)])
            self.module.fail_json(msg='Invalid controller %s HCA channel. Available channels: %s, Array Id [%s].' % (self.controller, status_msg, self.ssid))
        self.get_target_interface_cache = sorted_controller_ifaces[self.channel - 1][1]
    return self.get_target_interface_cache