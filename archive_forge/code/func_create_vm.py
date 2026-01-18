import functools
import time
import uuid
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import uuidutils
from six.moves import range  # noqa
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
from os_win.utils import pathutils
def create_vm(self, vm_name, vnuma_enabled, vm_gen, instance_path, notes=None):
    LOG.debug('Creating VM %s', vm_name)
    vs_data = self._compat_conn.Msvm_VirtualSystemSettingData.new()
    vs_data.ElementName = vm_name
    vs_data.Notes = notes
    vs_data.AutomaticStartupAction = self._AUTOMATIC_STARTUP_ACTION_NONE
    vs_data.VirtualNumaEnabled = vnuma_enabled
    if vm_gen == constants.VM_GEN_2:
        vs_data.VirtualSystemSubType = self._VIRTUAL_SYSTEM_SUBTYPE_GEN2
        vs_data.SecureBootEnabled = False
    vs_data.ConfigurationDataRoot = instance_path
    vs_data.LogDataRoot = instance_path
    vs_data.SnapshotDataRoot = instance_path
    vs_data.SuspendDataRoot = instance_path
    vs_data.SwapFileDataRoot = instance_path
    job_path, vm_path, ret_val = self._vs_man_svc.DefineSystem(ResourceSettings=[], ReferenceConfiguration=None, SystemSettings=vs_data.GetText_(1))
    self._jobutils.check_ret_val(ret_val, job_path)