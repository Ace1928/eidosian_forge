import base64
import re
from urllib.parse import urlparse, urlunparse
from ... import bedding, branch, errors, osutils, trace, transport
from ...i18n import gettext
from launchpadlib.credentials import (AccessToken, Credentials,
from launchpadlib.launchpad import Launchpad
def do_load(self, unique_key):
    """Retrieve credentials from the keyring."""
    auth_def = self.auth_config._get_config().get(unique_key)
    if auth_def and auth_def.get('access_secret'):
        access_token = AccessToken(auth_def.get('access_token'), auth_def.get('access_secret'))
        return Credentials(consumer_name=auth_def.get('consumer_key'), consumer_secret=auth_def.get('consumer_secret'), access_token=access_token, application_name='Breezy')
    return None