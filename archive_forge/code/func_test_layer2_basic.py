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
def test_layer2_basic(self):
    print('--- running Amazon DynamoDB Layer2 tests ---')
    c = self.dynamodb
    schema = c.create_schema(self.hash_key_name, self.hash_key_proto_value, self.range_key_name, self.range_key_proto_value)
    schema2 = c.create_schema('post_id', '')
    index = int(time.time())
    table_name = 'test-%d' % index
    read_units = 5
    write_units = 5
    table = self.create_table(table_name, schema, read_units, write_units)
    assert table.name == table_name
    assert table.schema.hash_key_name == self.hash_key_name
    assert table.schema.hash_key_type == get_dynamodb_type(self.hash_key_proto_value)
    assert table.schema.range_key_name == self.range_key_name
    assert table.schema.range_key_type == get_dynamodb_type(self.range_key_proto_value)
    assert table.read_units == read_units
    assert table.write_units == write_units
    assert table.item_count == 0
    assert table.size_bytes == 0
    table2_name = 'test-%d' % (index + 1)
    table2 = self.create_table(table2_name, schema2, read_units, write_units)
    table.refresh(wait_for_active=True)
    table2.refresh(wait_for_active=True)
    table_names = c.list_tables()
    assert table_name in table_names
    assert table2_name in table_names
    new_read_units = 10
    new_write_units = 5
    table.update_throughput(new_read_units, new_write_units)
    table.refresh(wait_for_active=True)
    assert table.read_units == new_read_units
    assert table.write_units == new_write_units
    item1_key = 'Amazon DynamoDB'
    item1_range = 'DynamoDB Thread 1'
    item1_attrs = {'Message': 'DynamoDB thread 1 message text', 'LastPostedBy': 'User A', 'Views': 0, 'Replies': 0, 'Answered': 0, 'Public': True, 'Tags': set(['index', 'primarykey', 'table']), 'LastPostDateTime': '12/9/2011 11:36:03 PM'}
    item1_attrs[self.hash_key_name] = 'foo'
    foobar_item = table.new_item(item1_key, item1_range, item1_attrs)
    assert foobar_item.hash_key == item1_key
    item1_attrs[self.range_key_name] = 'bar'
    foobar_item = table.new_item(item1_key, item1_range, item1_attrs)
    assert foobar_item.range_key == item1_range
    foobar_item = table.new_item(attrs=item1_attrs)
    assert foobar_item.hash_key == 'foo'
    assert foobar_item.range_key == 'bar'
    del item1_attrs[self.hash_key_name]
    del item1_attrs[self.range_key_name]
    item1 = table.new_item(item1_key, item1_range, item1_attrs)
    try:
        item1.put()
    except c.layer1.ResponseError as e:
        raise Exception('Item put failed: %s' % e)
    self.assertRaises(DynamoDBKeyNotFoundError, table.get_item, 'bogus_key', item1_range)
    item1_copy = table.get_item(item1_key, item1_range, consistent_read=True)
    assert item1_copy.hash_key == item1.hash_key
    assert item1_copy.range_key == item1.range_key
    for attr_name in item1_attrs:
        val = item1_copy[attr_name]
        if isinstance(val, (int, long_type, float, six.string_types)):
            assert val == item1[attr_name]
    attributes = ['Message', 'Views']
    item1_small = table.get_item(item1_key, item1_range, attributes_to_get=attributes, consistent_read=True)
    for attr_name in item1_small:
        if attr_name not in (item1_small.hash_key_name, item1_small.range_key_name):
            assert attr_name in attributes
    self.assertTrue(table.has_item(item1_key, range_key=item1_range, consistent_read=True))
    expected = {'Views': 1}
    self.assertRaises(DynamoDBConditionalCheckFailedError, item1.delete, expected_value=expected)
    expected = {'FooBar': True}
    try:
        item1.delete(expected_value=expected)
    except c.layer1.ResponseError:
        pass
    item1.add_attribute('Replies', 2)
    removed_attr = 'Public'
    item1.delete_attribute(removed_attr)
    removed_tag = item1_attrs['Tags'].copy().pop()
    item1.delete_attribute('Tags', set([removed_tag]))
    replies_by_set = set(['Adam', 'Arnie'])
    item1.put_attribute('RepliesBy', replies_by_set)
    retvals = item1.save(return_values='ALL_OLD')
    assert 'Attributes' in retvals
    item1_updated = table.get_item(item1_key, item1_range, consistent_read=True)
    assert item1_updated['Replies'] == item1_attrs['Replies'] + 2
    self.assertFalse(removed_attr in item1_updated)
    self.assertTrue(removed_tag not in item1_updated['Tags'])
    self.assertTrue('RepliesBy' in item1_updated)
    self.assertTrue(item1_updated['RepliesBy'] == replies_by_set)
    item2_key = 'Amazon DynamoDB'
    item2_range = 'DynamoDB Thread 2'
    item2_attrs = {'Message': 'DynamoDB thread 2 message text', 'LastPostedBy': 'User A', 'Views': 0, 'Replies': 0, 'Answered': 0, 'Tags': set(['index', 'primarykey', 'table']), 'LastPost2DateTime': '12/9/2011 11:36:03 PM'}
    item2 = table.new_item(item2_key, item2_range, item2_attrs)
    item2.put()
    item3_key = 'Amazon S3'
    item3_range = 'S3 Thread 1'
    item3_attrs = {'Message': 'S3 Thread 1 message text', 'LastPostedBy': 'User A', 'Views': 0, 'Replies': 0, 'Answered': 0, 'Tags': set(['largeobject', 'multipart upload']), 'LastPostDateTime': '12/9/2011 11:36:03 PM'}
    item3 = table.new_item(item3_key, item3_range, item3_attrs)
    item3.put()
    table2_item1_key = uuid.uuid4().hex
    table2_item1_attrs = {'DateTimePosted': '25/1/2011 12:34:56 PM', 'Text': 'I think boto rocks and so does DynamoDB'}
    table2_item1 = table2.new_item(table2_item1_key, attrs=table2_item1_attrs)
    table2_item1.put()
    items = table.query('Amazon DynamoDB', range_key_condition=BEGINS_WITH('DynamoDB'))
    n = 0
    for item in items:
        n += 1
    assert n == 2
    assert items.consumed_units > 0
    items = table.query('Amazon DynamoDB', range_key_condition=BEGINS_WITH('DynamoDB'), request_limit=1, max_results=1)
    n = 0
    for item in items:
        n += 1
    assert n == 1
    assert items.consumed_units > 0
    items = table.scan()
    n = 0
    for item in items:
        n += 1
    assert n == 3
    assert items.consumed_units > 0
    items = table.scan(scan_filter={'Replies': GT(0)})
    n = 0
    for item in items:
        n += 1
    assert n == 1
    assert items.consumed_units > 0
    integer_value = 42
    float_value = 345.678
    item3['IntAttr'] = integer_value
    item3['FloatAttr'] = float_value
    item3['TrueBoolean'] = True
    item3['FalseBoolean'] = False
    integer_set = set([1, 2, 3, 4, 5])
    float_set = set([1.1, 2.2, 3.3, 4.4, 5.5])
    mixed_set = set([1, 2, 3.3, 4, 5.555])
    str_set = set(['foo', 'bar', 'fie', 'baz'])
    item3['IntSetAttr'] = integer_set
    item3['FloatSetAttr'] = float_set
    item3['MixedSetAttr'] = mixed_set
    item3['StrSetAttr'] = str_set
    item3.put()
    item4 = table.get_item(item3_key, item3_range, consistent_read=True)
    assert item4['IntAttr'] == integer_value
    assert item4['FloatAttr'] == float_value
    assert bool(item4['TrueBoolean']) is True
    assert bool(item4['FalseBoolean']) is False
    for i in item4['IntSetAttr']:
        assert i in integer_set
    for i in item4['FloatSetAttr']:
        assert i in float_set
    for i in item4['MixedSetAttr']:
        assert i in mixed_set
    for i in item4['StrSetAttr']:
        assert i in str_set
    batch_list = c.new_batch_list()
    batch_list.add_batch(table, [(item2_key, item2_range), (item3_key, item3_range)])
    response = batch_list.submit()
    assert len(response['Responses'][table.name]['Items']) == 2
    batch_list = c.new_batch_list()
    batch_list.add_batch(table, [])
    response = batch_list.submit()
    assert response == {}
    item4_key = 'Amazon S3'
    item4_range = 'S3 Thread 2'
    item4_attrs = {'Message': 'S3 Thread 2 message text', 'LastPostedBy': 'User A', 'Views': 0, 'Replies': 0, 'Answered': 0, 'Tags': set(['largeobject', 'multipart upload']), 'LastPostDateTime': '12/9/2011 11:36:03 PM'}
    item5_key = 'Amazon S3'
    item5_range = 'S3 Thread 3'
    item5_attrs = {'Message': 'S3 Thread 3 message text', 'LastPostedBy': 'User A', 'Views': 0, 'Replies': 0, 'Answered': 0, 'Tags': set(['largeobject', 'multipart upload']), 'LastPostDateTime': '12/9/2011 11:36:03 PM'}
    item4 = table.new_item(item4_key, item4_range, item4_attrs)
    item5 = table.new_item(item5_key, item5_range, item5_attrs)
    batch_list = c.new_batch_write_list()
    batch_list.add_batch(table, puts=[item4, item5])
    response = batch_list.submit()
    results = table.scan(scan_filter={'Tags': CONTAINS('table')})
    assert results.scanned_count == 5
    results = table.scan(request_limit=2, max_results=5)
    assert results.count == 2
    for item in results:
        if results.count == 2:
            assert results.remaining == 4
            results.remaining -= 2
            results.next_response()
        else:
            assert results.count == 4
            assert results.remaining in (0, 1)
    assert results.count == 4
    results = table.scan(request_limit=6, max_results=4)
    assert len(list(results)) == 4
    assert results.count == 4
    batch_list = c.new_batch_write_list()
    batch_list.add_batch(table, deletes=[(item4_key, item4_range), (item5_key, item5_range)])
    response = batch_list.submit()
    results = table.query('Amazon DynamoDB', range_key_condition=BEGINS_WITH('DynamoDB'))
    n = 0
    for item in results:
        n += 1
    assert n == 2
    expected = {'Views': 0}
    item1.delete(expected_value=expected)
    self.assertFalse(table.has_item(item1_key, range_key=item1_range, consistent_read=True))
    ret_vals = item2.delete(return_values='ALL_OLD')
    assert ret_vals['Attributes'][self.hash_key_name] == item2_key
    assert ret_vals['Attributes'][self.range_key_name] == item2_range
    item3.delete()
    table2_item1.delete()
    print('--- tests completed ---')