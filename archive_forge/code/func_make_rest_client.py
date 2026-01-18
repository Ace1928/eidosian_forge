import sys
import pbr.version
from os_client_config import cloud_config
from os_client_config.config import OpenStackConfig  # noqa
from os_client_config import vendors  # noqa
def make_rest_client(service_key, options=None, app_name=None, app_version=None, version=None, **kwargs):
    """Simple wrapper function. It has almost no features.

    This will get you a raw requests Session Adapter that is mounted
    on the given service from the keystone service catalog. If you leave
    off cloud and region_name, it will assume that you've got env vars
    set, but if you give them, it'll use clouds.yaml as you'd expect.

    This function is deliberately simple. It has no flexibility. If you
    want flexibility, you can make a cloud config object and call
    get_session_client on it. This function is to make it easy to poke
    at OpenStack REST APIs with a properly configured keystone session.
    """
    cloud = get_config(service_key=service_key, options=options, app_name=app_name, app_version=app_version, **kwargs)
    return cloud.get_session_client(service_key, version=version)