import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def get_class_for_auth_version(auth_version):
    """
    Retrieve class for the provided auth version.
    """
    if auth_version == '1.0':
        cls = OpenStackIdentity_1_0_Connection
    elif auth_version == '1.1':
        cls = OpenStackIdentity_1_1_Connection
    elif auth_version == '2.0' or auth_version == '2.0_apikey':
        cls = OpenStackIdentity_2_0_Connection
    elif auth_version == '2.0_password':
        cls = OpenStackIdentity_2_0_Connection
    elif auth_version == '2.0_voms':
        cls = OpenStackIdentity_2_0_Connection_VOMS
    elif auth_version == '3.x_password':
        cls = OpenStackIdentity_3_0_Connection
    elif auth_version == '3.x_appcred':
        cls = OpenStackIdentity_3_0_Connection_AppCred
    elif auth_version == '3.x_oidc_access_token':
        cls = OpenStackIdentity_3_0_Connection_OIDC_access_token
    else:
        raise LibcloudError('Unsupported Auth Version requested: %s' % auth_version)
    return cls