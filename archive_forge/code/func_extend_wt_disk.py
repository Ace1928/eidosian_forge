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
def extend_wt_disk(self, wtd_name, additional_mb):
    try:
        wt_disk = self._get_wt_disk(wtd_name)
        wt_disk.Extend(additional_mb)
    except exceptions.x_wmi as wmi_exc:
        err_msg = _('Could not extend WT Disk %(wtd_name)s with additional %(additional_mb)s MB.')
        raise exceptions.ISCSITargetWMIException(err_msg % dict(wtd_name=wtd_name, additional_mb=additional_mb), wmi_exc=wmi_exc)