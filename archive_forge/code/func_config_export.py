from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def config_export(self):
    """configure sflow export"""
    xml_str = ''
    if not self.export_route:
        return xml_str
    if self.export_route == 'enable':
        if self.sflow_dict['export'] and self.sflow_dict['export'].get('ExportRoute') == 'disable':
            xml_str = '<exports><export operation="delete"><ExportRoute>disable</ExportRoute></export></exports>'
            self.updates_cmd.append('undo sflow export extended-route-data disable')
    elif not self.sflow_dict['export'] or self.sflow_dict['export'].get('ExportRoute') != 'disable':
        xml_str = '<exports><export operation="create"><ExportRoute>disable</ExportRoute></export></exports>'
        self.updates_cmd.append('sflow export extended-route-data disable')
    return xml_str