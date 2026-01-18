import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kms import exceptions
from boto.compat import six
import base64
def put_key_policy(self, key_id, policy_name, policy):
    """
        Attaches a policy to the specified key.

        :type key_id: string
        :param key_id: Unique identifier of the key. This can be an ARN, an
            alias, or a globally unique identifier.

        :type policy_name: string
        :param policy_name: Name of the policy to be attached. Currently, the
            only supported name is "default".

        :type policy: string
        :param policy: The policy, in JSON format, to be attached to the key.

        """
    params = {'KeyId': key_id, 'PolicyName': policy_name, 'Policy': policy}
    return self.make_request(action='PutKeyPolicy', body=json.dumps(params))