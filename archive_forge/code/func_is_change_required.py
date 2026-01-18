from __future__ import absolute_import, division, print_function
import re
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
def is_change_required(self):
    """Determine whether change is required."""
    changed_required = False
    iface = self.get_target_interface()
    if iface['ioInterfaceTypeData']['interfaceType'] == 'iscsi' and iface['ioInterfaceTypeData']['iscsi']['ipv4Data']['ipv4AddressData']['ipv4Address'] != self.address:
        changed_required = True
    elif iface['ioInterfaceTypeData']['interfaceType'] == 'ib' and iface['ioInterfaceTypeData']['ib']['isISERSupported']:
        for properties in iface['commandProtocolPropertiesList']['commandProtocolProperties']:
            if properties['commandProtocol'] == 'scsi' and properties['scsiProperties']['scsiProtocolType'] == 'iser' and (properties['scsiProperties']['iserProperties']['ipv4Data']['ipv4AddressData']['ipv4Address'] != self.address):
                changed_required = True
    return changed_required