from unittest import mock
from keystoneauth1 import identity
from keystoneauth1 import session
def get_server_supported_versions(min_version, max_version):
    if min_version and max_version:
        return get_custom_current_response(min_version, max_version)
    return STABLE_RESPONSE