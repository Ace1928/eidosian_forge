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
def generate_presigned_url(self, url, date_less_than=None, policy=None):
    """Creates a signed CloudFront URL based on given parameters.

        :type url: str
        :param url: The URL of the protected object

        :type date_less_than: datetime
        :param date_less_than: The URL will expire after that date and time

        :type policy: str
        :param policy: The custom policy, possibly built by self.build_policy()

        :rtype: str
        :return: The signed URL.
        """
    both_args_supplied = date_less_than is not None and policy is not None
    neither_arg_supplied = date_less_than is None and policy is None
    if both_args_supplied or neither_arg_supplied:
        e = 'Need to provide either date_less_than or policy, but not both'
        raise ValueError(e)
    if date_less_than is not None:
        policy = self.build_policy(url, date_less_than)
    if isinstance(policy, str):
        policy = policy.encode('utf8')
    if date_less_than is not None:
        params = ['Expires=%s' % int(datetime2timestamp(date_less_than))]
    else:
        params = ['Policy=%s' % self._url_b64encode(policy).decode('utf8')]
    signature = self.rsa_signer(policy)
    params.extend([f'Signature={self._url_b64encode(signature).decode('utf8')}', f'Key-Pair-Id={self.key_id}'])
    return self._build_url(url, params)