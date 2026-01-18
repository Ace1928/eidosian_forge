import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def get_nic_hardware_offload_info(self):
    """Get host's NIC hardware offload information.

        Hyper-V offers a few different hardware offloading options for VMs and
        their vNICs, depending on the vSwitches' NICs hardware resources and
        capabilities. These resources are managed and assigned automatically by
        Hyper-V. These resources are: VFs, IOV queue pairs, VMQs, IPsec
        security association offloads.

        :returns: a list of dictionaries, containing the following fields:
            - 'vswitch_name': the switch name.
            - 'device_id': the switch's physical NIC's PnP device ID.
            - 'total_vfs': the switch's maximum number of VFs. (>= 0)
            - 'used_vfs': the switch's number of used VFs. (<= 'total_vfs')
            - 'total_iov_queue_pairs': the switch's maximum number of IOV
                queue pairs. (>= 'total_vfs')
            - 'used_iov_queue_pairs': the switch's number of used IOV queue
                pairs (<= 'total_iov_queue_pairs')
            - 'total_vmqs': the switch's maximum number of VMQs. (>= 0)
            - 'used_vmqs': the switch's number of used VMQs. (<= 'total_vmqs')
            - 'total_ipsecsa': the maximum number of IPsec SA offloads
                supported by the switch. (>= 0)
            - 'used_ipsecsa': the switch's number of IPsec SA offloads
                currently in use. (<= 'total_ipsecsa')
        """
    hw_offload_data = []
    vswitch_sds = self._conn.Msvm_VirtualEthernetSwitchSettingData()
    hw_offload_sds = self._conn.Msvm_EthernetSwitchHardwareOffloadData()
    for vswitch_sd in vswitch_sds:
        hw_offload = [s for s in hw_offload_sds if s.SystemName == vswitch_sd.VirtualSystemIdentifier][0]
        vswitch_offload_data = self._get_nic_hw_offload_info(vswitch_sd, hw_offload)
        if vswitch_offload_data:
            hw_offload_data.append(vswitch_offload_data)
    return hw_offload_data