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
def check_version_deprecated(api_version):
    """Returns True if API version is deprecated."""
    if api_version == manilaclient.API_DEPRECATED_VERSION:
        msg = _("Client version '%(version)s' is deprecated.") % {'version': api_version.get_string()}
        warnings.warn(msg)
        return True
    return False