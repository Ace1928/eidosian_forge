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
def disable_remotefx_video_adapter(self, vm_name):
    vm = self._lookup_vm_check(vm_name)
    rasds = _wqlutils.get_element_associated_class(self._compat_conn, self._CIM_RES_ALLOC_SETTING_DATA_CLASS, element_instance_id=vm.InstanceID)
    remotefx_disp_ctrl_res = [r for r in rasds if r.ResourceSubType == self._REMOTEFX_DISP_CTRL_RES_SUB_TYPE]
    if not remotefx_disp_ctrl_res:
        return
    self._jobutils.remove_virt_resource(remotefx_disp_ctrl_res[0])
    synth_disp_ctrl_res = self._get_new_resource_setting_data(self._SYNTH_DISP_CTRL_RES_SUB_TYPE, self._SYNTH_DISP_ALLOCATION_SETTING_DATA_CLASS)
    self._jobutils.add_virt_resource(synth_disp_ctrl_res, vm)
    if self._vm_has_s3_controller(vm_name):
        s3_disp_ctrl_res = [r for r in rasds if r.ResourceSubType == self._S3_DISP_CTRL_RES_SUB_TYPE][0]
        s3_disp_ctrl_res.Address = self._DISP_CTRL_ADDRESS
        self._jobutils.modify_virt_resource(s3_disp_ctrl_res)