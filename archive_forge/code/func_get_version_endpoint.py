from unittest import mock
from keystoneauth1 import identity
from keystoneauth1 import session
def get_version_endpoint(endpoint=None):
    return '{}/v1/'.format(endpoint or _DEFAULT_ENDPOINT)