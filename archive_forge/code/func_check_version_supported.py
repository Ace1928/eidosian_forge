import functools
import logging
import re
import warnings
import manilaclient
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient import utils
def check_version_supported(api_version):
    """Returns True if the API version is supported.

    :warn Sends warning if version is not supported.
    """
    if check_version_matches_min_max(api_version) or check_version_deprecated(api_version):
        return True
    return False