import os
import re
import warnings
from os_win import utilsfactory
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.remotefs import remotefs
def get_share_subdir(self, share):
    return '\\'.join(self._get_share_norm_path(share).lstrip('\\').split('\\')[2:])