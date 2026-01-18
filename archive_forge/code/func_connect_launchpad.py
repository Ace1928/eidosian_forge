import base64
import re
from urllib.parse import urlparse, urlunparse
from ... import bedding, branch, errors, osutils, trace, transport
from ...i18n import gettext
from launchpadlib.credentials import (AccessToken, Credentials,
from launchpadlib.launchpad import Launchpad
def connect_launchpad(base_url, timeout=None, proxy_info=None, version=Launchpad.DEFAULT_VERSION):
    """Log in to the Launchpad API.

    :return: The root `Launchpad` object from launchpadlib.
    """
    if proxy_info is None:
        import httplib2
        proxy_info = httplib2.proxy_info_from_environment('https')
    try:
        cache_directory = get_cache_directory()
    except OSError:
        cache_directory = None
    credential_store = get_credential_store()
    authorization_engine = get_auth_engine(base_url)
    from .account import get_lp_login
    lp_user = get_lp_login()
    if lp_user is None:
        trace.mutter('Accessing launchpad API anonymously, since no username is set.')
        return Launchpad.login_anonymously(consumer_name='breezy', service_root=base_url, launchpadlib_dir=cache_directory, timeout=timeout, proxy_info=proxy_info, version=version)
    else:
        return Launchpad.login_with(application_name='breezy', service_root=base_url, launchpadlib_dir=cache_directory, timeout=timeout, credential_store=credential_store, authorization_engine=authorization_engine, proxy_info=proxy_info, version=version)