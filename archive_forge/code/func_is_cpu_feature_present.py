import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def is_cpu_feature_present(self, feature_key):
    """Checks if the host's CPUs have the given feature."""
    return kernel32.IsProcessorFeaturePresent(feature_key)