import os
import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import win32utils
def check_smb_mapping(self, share_path, remove_unavailable_mapping=False):
    mappings = self._smb_conn.Msft_SmbMapping(RemotePath=share_path)
    if not mappings:
        return False
    if os.path.exists(share_path):
        LOG.debug('Share already mounted: %s', share_path)
        return True
    else:
        LOG.debug('Share exists but is unavailable: %s ', share_path)
        if remove_unavailable_mapping:
            self.unmount_smb_share(share_path, force=True)
        return False