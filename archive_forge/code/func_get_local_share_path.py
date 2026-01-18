import os
import re
import warnings
from os_win import utilsfactory
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.remotefs import remotefs
def get_local_share_path(self, share, expect_existing=True):
    share = self._get_share_norm_path(share)
    share_name = self.get_share_name(share)
    share_subdir = self.get_share_subdir(share)
    is_local_share = self._smbutils.is_local_share(share)
    if not is_local_share:
        LOG.debug("Share '%s' is not exposed by the current host.", share)
        local_share_path = None
    else:
        local_share_path = self._smbutils.get_smb_share_path(share_name)
    if not local_share_path and expect_existing:
        err_msg = _('Could not find the local share path for %(share)s.')
        raise exception.VolumePathsNotFound(err_msg % dict(share=share))
    if local_share_path and share_subdir:
        local_share_path = os.path.join(local_share_path, share_subdir)
    return local_share_path