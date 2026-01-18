import time
from binascii import crc32
import boto
from boto.connection import AWSAuthConnection
from boto.exception import DynamoDBResponseError
from boto.provider import Provider
from boto.dynamodb import exceptions as dynamodb_exceptions
from boto.compat import json
def delete_table(self, table_name):
    """
        Deletes the table and all of it's data.  After this request
        the table will be in the DELETING state until DynamoDB
        completes the delete operation.

        :type table_name: str
        :param table_name: The name of the table to delete.
        """
    data = {'TableName': table_name}
    json_input = json.dumps(data)
    return self.make_request('DeleteTable', json_input)