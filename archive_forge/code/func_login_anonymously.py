import errno
import os
import warnings
from lazr.restfulclient.resource import (  # noqa: F401
from lazr.restfulclient.authorize.oauth import SystemWideConsumer
from lazr.restfulclient._browser import RestfulHttp
from launchpadlib.credentials import (
from launchpadlib import uris
from launchpadlib.uris import (  # noqa: F401
@classmethod
def login_anonymously(cls, consumer_name, service_root=uris.STAGING_SERVICE_ROOT, launchpadlib_dir=None, timeout=None, proxy_info=proxy_info_from_environment, version=DEFAULT_VERSION):
    """Get access to Launchpad without providing any credentials."""
    service_root, launchpadlib_dir, cache_path, service_root_dir = cls._get_paths(service_root, launchpadlib_dir)
    token = AnonymousAccessToken()
    credentials = Credentials(consumer_name, access_token=token)
    return cls(credentials, None, None, service_root=service_root, cache=cache_path, timeout=timeout, proxy_info=proxy_info, version=version)