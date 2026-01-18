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
def test_get_classic_link_instances_params(self):
    self.set_http_response(status_code=200)
    self.ec2.get_all_classic_link_instances(instance_ids=['id1', 'id2'], filters={'GroupId': 'sg-9b4343fe'}, dry_run=True, next_token='next_token', max_results=10)
    self.assert_request_parameters({'Action': 'DescribeClassicLinkInstances', 'InstanceId.1': 'id1', 'InstanceId.2': 'id2', 'Filter.1.Name': 'GroupId', 'Filter.1.Value.1': 'sg-9b4343fe', 'DryRun': 'true', 'NextToken': 'next_token', 'MaxResults': 10}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])