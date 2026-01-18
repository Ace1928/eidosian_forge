import base64
import re
from urllib.parse import urlparse, urlunparse
from ... import bedding, branch, errors, osutils, trace, transport
from ...i18n import gettext
from launchpadlib.credentials import (AccessToken, Credentials,
from launchpadlib.launchpad import Launchpad
def get_cache_directory():
    """Return the directory to cache launchpadlib objects in."""
    return osutils.pathjoin(bedding.cache_dir(), 'launchpad')