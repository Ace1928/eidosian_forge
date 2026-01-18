from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import hostutils
from os_win.utils import pathutils
from os_win.utils import win32utils
def remove_wt_disk(self, wtd_name):
    try:
        wt_disk = self._get_wt_disk(wtd_name, fail_if_not_found=False)
        if wt_disk:
            wt_disk.Delete_()
    except exceptions.x_wmi as wmi_exc:
        err_msg = _('Failed to remove WT disk: %s.')
        raise exceptions.ISCSITargetWMIException(err_msg % wtd_name, wmi_exc=wmi_exc)