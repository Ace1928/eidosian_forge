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
def generate_presigned_post(self, request_dict, fields=None, conditions=None, expires_in=3600, region_name=None):
    """Generates the url and the form fields used for a presigned s3 post

        :type request_dict: dict
        :param request_dict: The prepared request dictionary returned by
            ``botocore.awsrequest.prepare_request_dict()``

        :type fields: dict
        :param fields: A dictionary of prefilled form fields to build on top
            of.

        :type conditions: list
        :param conditions: A list of conditions to include in the policy. Each
            element can be either a list or a structure. For example:
            [
             {"acl": "public-read"},
             {"bucket": "mybucket"},
             ["starts-with", "$key", "mykey"]
            ]

        :type expires_in: int
        :param expires_in: The number of seconds the presigned post is valid
            for.

        :type region_name: string
        :param region_name: The region name to sign the presigned post to.

        :rtype: dict
        :returns: A dictionary with two elements: ``url`` and ``fields``.
            Url is the url to post to. Fields is a dictionary filled with
            the form fields and respective values to use when submitting the
            post. For example:

            {'url': 'https://mybucket.s3.amazonaws.com
             'fields': {'acl': 'public-read',
                        'key': 'mykey',
                        'signature': 'mysignature',
                        'policy': 'mybase64 encoded policy'}
            }
        """
    if fields is None:
        fields = {}
    if conditions is None:
        conditions = []
    policy = {}
    datetime_now = datetime.datetime.utcnow()
    expire_date = datetime_now + datetime.timedelta(seconds=expires_in)
    policy['expiration'] = expire_date.strftime(botocore.auth.ISO8601)
    policy['conditions'] = []
    for condition in conditions:
        policy['conditions'].append(condition)
    request = create_request_object(request_dict)
    request.context['s3-presign-post-fields'] = fields
    request.context['s3-presign-post-policy'] = policy
    self._request_signer.sign('PutObject', request, region_name, 'presign-post')
    return {'url': request.url, 'fields': fields}