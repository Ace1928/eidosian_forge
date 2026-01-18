import os
import time
from tests.unit import unittest
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey, KeysOnlyIndex,
from boto.dynamodb2.items import Item
from boto.dynamodb2.table import Table
from boto.dynamodb2.types import NUMBER, STRING
def test_gsi(self):
    users = Table.create('gsi_users', schema=[HashKey('user_id')], throughput={'read': 5, 'write': 3}, global_indexes=[GlobalKeysOnlyIndex('StuffIndex', parts=[HashKey('user_id')], throughput={'read': 2, 'write': 1})])
    self.addCleanup(users.delete)
    time.sleep(60)
    users.update(throughput={'read': 3, 'write': 4}, global_indexes={'StuffIndex': {'read': 1, 'write': 2}})
    time.sleep(150)