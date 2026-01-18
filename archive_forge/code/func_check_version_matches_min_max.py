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
def check_version_matches_min_max(api_version):
    """Returns True if the API version is within the supported range."""
    if not api_version.matches(manilaclient.API_MIN_VERSION, manilaclient.API_MAX_VERSION):
        msg = _("Invalid client version '%(version)s'. Current version range is '%(min)s' through  '%(max)s'") % {'version': api_version.get_string(), 'min': manilaclient.API_MIN_VERSION.get_string(), 'max': manilaclient.API_MAX_VERSION.get_string()}
        warnings.warn(msg)
        return False
    return True