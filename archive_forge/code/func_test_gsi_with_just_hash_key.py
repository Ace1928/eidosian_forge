import os
import time
from tests.unit import unittest
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey, KeysOnlyIndex,
from boto.dynamodb2.items import Item
from boto.dynamodb2.table import Table
from boto.dynamodb2.types import NUMBER, STRING
def test_gsi_with_just_hash_key(self):
    users = Table.create('gsi_query_users', schema=[HashKey('user_id')], throughput={'read': 5, 'write': 3}, global_indexes=[GlobalIncludeIndex('UsernameIndex', parts=[HashKey('username')], includes=['user_id', 'username'], throughput={'read': 3, 'write': 1})])
    self.addCleanup(users.delete)
    time.sleep(60)
    users.put_item(data={'user_id': '7', 'username': 'johndoe', 'first_name': 'John', 'last_name': 'Doe'})
    users.put_item(data={'user_id': '24', 'username': 'alice', 'first_name': 'Alice', 'last_name': 'Expert'})
    users.put_item(data={'user_id': '35', 'username': 'jane', 'first_name': 'Jane', 'last_name': 'Doe'})
    rs = users.query_2(user_id__eq='24')
    results = sorted([user['username'] for user in rs])
    self.assertEqual(results, ['alice'])
    rs = users.query_2(username__eq='johndoe', index='UsernameIndex')
    results = sorted([user['username'] for user in rs])
    self.assertEqual(results, ['johndoe'])