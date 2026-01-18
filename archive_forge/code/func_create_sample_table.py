import time
import uuid
from decimal import Decimal
from tests.unit import unittest
from boto.dynamodb.exceptions import DynamoDBKeyNotFoundError
from boto.dynamodb.exceptions import DynamoDBConditionalCheckFailedError
from boto.dynamodb.layer2 import Layer2
from boto.dynamodb.types import get_dynamodb_type, Binary
from boto.dynamodb.condition import BEGINS_WITH, CONTAINS, GT
from boto.compat import six, long_type
def create_sample_table(self):
    schema = self.dynamodb.create_schema(self.hash_key_name, self.hash_key_proto_value, self.range_key_name, self.range_key_proto_value)
    table = self.create_table(self.table_name, schema, 5, 5)
    table.refresh(wait_for_active=True)
    return table