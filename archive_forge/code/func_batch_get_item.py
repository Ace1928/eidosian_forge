import time
from binascii import crc32
import boto
from boto.connection import AWSAuthConnection
from boto.exception import DynamoDBResponseError
from boto.provider import Provider
from boto.dynamodb import exceptions as dynamodb_exceptions
from boto.compat import json
def batch_get_item(self, request_items, object_hook=None):
    """
        Return a set of attributes for a multiple items in
        multiple tables using their primary keys.

        :type request_items: dict
        :param request_items: A Python version of the RequestItems
            data structure defined by DynamoDB.
        """
    if not request_items:
        return {}
    data = {'RequestItems': request_items}
    json_input = json.dumps(data)
    return self.make_request('BatchGetItem', json_input, object_hook=object_hook)