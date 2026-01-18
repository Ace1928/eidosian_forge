import functools
import re
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win._i18n import _
from os_win import conf
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
def set_vswitch_port_offload(self, switch_port_name, sriov_enabled=None, iov_queues_requested=None, vmq_enabled=None, offloaded_sa=None):
    """Enables / Disables different offload options for the given port.

        Optional prameters are ignored if they are None.

        :param switch_port_name: the name of the port which will have VMQ
            enabled or disabled.
        :param sriov_enabled: if SR-IOV should be turned on or off.
        :param iov_queues_requested: the number of IOV queues to use. (> 1)
        :param vmq_enabled: if VMQ should be turned on or off.
        :param offloaded_sa: the number of IPsec SA offloads to use. (> 1)
        :raises os_win.exceptions.InvalidParameterValue: if an invalid value
            is passed for the iov_queues_requested or offloaded_sa parameters.
        """
    if iov_queues_requested is not None and iov_queues_requested < 1:
        raise exceptions.InvalidParameterValue(param_name='iov_queues_requested', param_value=iov_queues_requested)
    if offloaded_sa is not None and offloaded_sa < 1:
        raise exceptions.InvalidParameterValue(param_name='offloaded_sa', param_value=offloaded_sa)
    port_alloc = self._get_switch_port_allocation(switch_port_name)[0]
    hw_offload_sd = self._get_hw_offload_sd_from_port_alloc(port_alloc)
    sd_changed = False
    if sriov_enabled is not None:
        desired_state = self._OFFLOAD_ENABLED if sriov_enabled else self._OFFLOAD_DISABLED
        if hw_offload_sd.IOVOffloadWeight != desired_state:
            hw_offload_sd.IOVOffloadWeight = desired_state
            sd_changed = True
    if iov_queues_requested is not None:
        if hw_offload_sd.IOVQueuePairsRequested != iov_queues_requested:
            hw_offload_sd.IOVQueuePairsRequested = iov_queues_requested
            sd_changed = True
    if vmq_enabled is not None:
        desired_state = self._OFFLOAD_ENABLED if vmq_enabled else self._OFFLOAD_DISABLED
        if hw_offload_sd.VMQOffloadWeight != desired_state:
            hw_offload_sd.VMQOffloadWeight = desired_state
            sd_changed = True
    if offloaded_sa is not None:
        if hw_offload_sd.IPSecOffloadLimit != offloaded_sa:
            hw_offload_sd.IPSecOffloadLimit = offloaded_sa
            sd_changed = True
    if sd_changed:
        self._jobutils.modify_virt_feature(hw_offload_sd)