import time
from binascii import crc32
import boto
from boto.connection import AWSAuthConnection
from boto.exception import DynamoDBResponseError
from boto.provider import Provider
from boto.dynamodb import exceptions as dynamodb_exceptions
from boto.compat import json
def batch_write_item(self, request_items, object_hook=None):
    """
        This operation enables you to put or delete several items
        across multiple tables in a single API call.

        :type request_items: dict
        :param request_items: A Python version of the RequestItems
            data structure defined by DynamoDB.
        """
    data = {'RequestItems': request_items}
    json_input = json.dumps(data)
    return self.make_request('BatchWriteItem', json_input, object_hook=object_hook)