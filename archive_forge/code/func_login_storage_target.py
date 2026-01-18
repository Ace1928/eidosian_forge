import ctypes
import functools
import inspect
import socket
import time
from oslo_log import log as logging
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
def login_storage_target(self, target_lun, target_iqn, target_portal, auth_username=None, auth_password=None, auth_type=None, mpio_enabled=False, ensure_lun_available=True, initiator_name=None, rescan_attempts=_DEFAULT_RESCAN_ATTEMPTS):
    portal_addr, portal_port = _utils.parse_server_string(target_portal)
    portal_port = int(portal_port) if portal_port else self._DEFAULT_ISCSI_PORT
    known_targets = self.get_targets()
    if target_iqn not in known_targets:
        self._add_static_target(target_iqn)
    login_required = self._new_session_required(target_iqn, portal_addr, portal_port, initiator_name, mpio_enabled)
    if login_required:
        LOG.debug('Logging in iSCSI target %(target_iqn)s', dict(target_iqn=target_iqn))
        login_flags = w_const.ISCSI_LOGIN_FLAG_MULTIPATH_ENABLED if mpio_enabled else 0
        login_opts = self._get_login_opts(auth_username, auth_password, auth_type, login_flags)
        portal = iscsi_struct.ISCSI_TARGET_PORTAL(Address=portal_addr, Socket=portal_port)
        self._login_iscsi_target(target_iqn, portal, login_opts, is_persistent=True)
        sid, cid = self._login_iscsi_target(target_iqn, portal, login_opts, is_persistent=False)
    if ensure_lun_available:
        self.ensure_lun_available(target_iqn, target_lun, rescan_attempts)