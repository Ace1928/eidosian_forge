import os
from tests.unit import unittest
def test_connect_to_unkown_region(self):
    from boto.dynamodb2 import connect_to_region
    from boto.dynamodb2.layer1 import DynamoDBConnection
    os.environ['BOTO_USE_ENDPOINT_HEURISTICS'] = 'True'
    connection = connect_to_region('us-east-45', aws_access_key_id='foo', aws_secret_access_key='bar')
    self.assertIsInstance(connection, DynamoDBConnection)
    self.assertEqual(connection.host, 'dynamodb.us-east-45.amazonaws.com')