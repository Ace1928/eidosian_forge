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
def get_portal_locations(self, available_only=True, fail_if_none_found=True):
    wt_portals = self._conn_wmi.WT_Portal()
    if available_only:
        wt_portals = list(filter(lambda portal: portal.Listen, wt_portals))
    if not wt_portals and fail_if_none_found:
        err_msg = _('No valid iSCSI portal was found.')
        raise exceptions.ISCSITargetException(err_msg)
    portal_locations = [self._get_portal_location(portal) for portal in wt_portals]
    return portal_locations