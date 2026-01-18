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
def get_supported_vhd_type(self):
    return constants.VHD_TYPE_DYNAMIC if self._win_gteq_6_3 else constants.VHD_TYPE_FIXED