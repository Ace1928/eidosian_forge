import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kms import exceptions
from boto.compat import six
import base64
def create_alias(self, alias_name, target_key_id):
    """
        Creates a display name for a customer master key. An alias can
        be used to identify a key and should be unique. The console
        enforces a one-to-one mapping between the alias and a key. An
        alias name can contain only alphanumeric characters, forward
        slashes (/), underscores (_), and dashes (-). An alias must
        start with the word "alias" followed by a forward slash
        (alias/). An alias that begins with "aws" after the forward
        slash (alias/aws...) is reserved by Amazon Web Services (AWS).

        :type alias_name: string
        :param alias_name: String that contains the display name. Aliases that
            begin with AWS are reserved.

        :type target_key_id: string
        :param target_key_id: An identifier of the key for which you are
            creating the alias. This value cannot be another alias.

        """
    params = {'AliasName': alias_name, 'TargetKeyId': target_key_id}
    return self.make_request(action='CreateAlias', body=json.dumps(params))