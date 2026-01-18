import base64
import datetime
import json
import weakref
import botocore
import botocore.auth
from botocore.awsrequest import create_request_object, prepare_request_dict
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import ArnParser, datetime2timestamp
from botocore.utils import fix_s3_host  # noqa
def generate_db_auth_token(self, DBHostname, Port, DBUsername, Region=None):
    """Generates an auth token used to connect to a db with IAM credentials.

    :type DBHostname: str
    :param DBHostname: The hostname of the database to connect to.

    :type Port: int
    :param Port: The port number the database is listening on.

    :type DBUsername: str
    :param DBUsername: The username to log in as.

    :type Region: str
    :param Region: The region the database is in. If None, the client
        region will be used.

    :return: A presigned url which can be used as an auth token.
    """
    region = Region
    if region is None:
        region = self.meta.region_name
    params = {'Action': 'connect', 'DBUser': DBUsername}
    request_dict = {'url_path': '/', 'query_string': '', 'headers': {}, 'body': params, 'method': 'GET'}
    scheme = 'https://'
    endpoint_url = f'{scheme}{DBHostname}:{Port}'
    prepare_request_dict(request_dict, endpoint_url)
    presigned_url = self._request_signer.generate_presigned_url(operation_name='connect', request_dict=request_dict, region_name=region, expires_in=900, signing_name='rds-db')
    return presigned_url[len(scheme):]