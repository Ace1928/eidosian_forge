from datetime import datetime, timedelta
from mock import MagicMock, Mock
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
import boto.ec2
from boto.regioninfo import RegionInfo
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from boto.ec2.connection import EC2Connection
from boto.ec2.snapshot import Snapshot
from boto.ec2.reservedinstance import ReservedInstancesConfiguration
from boto.compat import http_client
def test_create_reserved_instances_listing(self):
    self.set_http_response(status_code=200)
    response = self.ec2.create_reserved_instances_listing('instance_id', 1, [('2.5', 11), ('2.0', 8)], 'client_token')
    self.assertEqual(len(response), 1)
    cancellation = response[0]
    self.assertEqual(cancellation.status, 'active')
    self.assertEqual(cancellation.status_message, 'ACTIVE')
    self.assertEqual(len(cancellation.instance_counts), 4)
    first = cancellation.instance_counts[0]
    self.assertEqual(first.state, 'Available')
    self.assertEqual(first.instance_count, 1)
    self.assertEqual(len(cancellation.price_schedules), 11)
    schedule = cancellation.price_schedules[0]
    self.assertEqual(schedule.term, 11)
    self.assertEqual(schedule.price, '2.5')
    self.assertEqual(schedule.currency_code, 'USD')
    self.assertEqual(schedule.active, True)
    self.assert_request_parameters({'Action': 'CreateReservedInstancesListing', 'ReservedInstancesId': 'instance_id', 'InstanceCount': '1', 'ClientToken': 'client_token', 'PriceSchedules.0.Price': '2.5', 'PriceSchedules.0.Term': '11', 'PriceSchedules.1.Price': '2.0', 'PriceSchedules.1.Term': '8'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])