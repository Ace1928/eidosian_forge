import contextlib
import ctypes
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
import os_win.conf
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import hbaapi as fc_struct
def get_fc_target_mappings(self, node_wwn):
    """Retrieve FCP target mappings.

        :param node_wwn: a HBA node WWN represented as a hex string.
        :returns: a list of FCP mappings represented as dicts.
        """
    mappings = []
    node_wwn_struct = self._wwn_struct_from_hex_str(node_wwn)
    with self._get_hba_handle(adapter_wwn_struct=node_wwn_struct) as hba_handle:
        fcp_mappings = self._get_target_mapping(hba_handle)
        for entry in fcp_mappings.Entries:
            wwnn = _utils.byte_array_to_hex_str(entry.FcpId.NodeWWN.wwn)
            wwpn = _utils.byte_array_to_hex_str(entry.FcpId.PortWWN.wwn)
            mapping = dict(node_name=wwnn, port_name=wwpn, device_name=entry.ScsiId.OSDeviceName, lun=entry.ScsiId.ScsiOSLun, fcp_lun=entry.FcpId.FcpLun)
            mappings.append(mapping)
    return mappings