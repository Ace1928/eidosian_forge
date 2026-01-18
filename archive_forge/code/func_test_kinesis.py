import time
import boto
from tests.compat import unittest
from boto.kinesis.exceptions import ResourceNotFoundException
def test_kinesis(self):
    kinesis = self.kinesis
    kinesis.create_stream('test', 1)
    self.addCleanup(self.kinesis.delete_stream, 'test')
    tries = 0
    while tries < 10:
        tries += 1
        time.sleep(15)
        response = kinesis.describe_stream('test')
        if response['StreamDescription']['StreamStatus'] == 'ACTIVE':
            shard_id = response['StreamDescription']['Shards'][0]['ShardId']
            break
    else:
        raise TimeoutError('Stream is still not active, aborting...')
    kinesis.add_tags_to_stream(stream_name='test', tags={'foo': 'bar'})
    response = kinesis.list_tags_for_stream(stream_name='test')
    self.assertEqual(len(response['Tags']), 1)
    self.assertEqual(response['Tags'][0], {'Key': 'foo', 'Value': 'bar'})
    kinesis.remove_tags_from_stream(stream_name='test', tag_keys=['foo'])
    response = kinesis.list_tags_for_stream(stream_name='test')
    self.assertEqual(len(response['Tags']), 0)
    response = kinesis.get_shard_iterator('test', shard_id, 'TRIM_HORIZON')
    shard_iterator = response['ShardIterator']
    data = 'Some data ...'
    record = {'Data': data, 'PartitionKey': data}
    response = kinesis.put_record('test', data, data)
    response = kinesis.put_records([record, record.copy()], 'test')
    tries = 0
    num_collected = 0
    num_expected_records = 3
    collected_records = []
    while tries < 100:
        tries += 1
        time.sleep(1)
        response = kinesis.get_records(shard_iterator)
        shard_iterator = response['NextShardIterator']
        for record in response['Records']:
            if 'Data' in record:
                collected_records.append(record['Data'])
                num_collected += 1
        if num_collected >= num_expected_records:
            self.assertEqual(num_expected_records, num_collected)
            break
    else:
        raise TimeoutError('No records found, aborting...')
    for record in collected_records:
        self.assertEqual(data, record)