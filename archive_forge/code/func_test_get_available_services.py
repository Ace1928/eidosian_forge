import mock
import os
import json
from nose.tools import assert_equal
from tests.unit import unittest
import boto
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
def test_get_available_services(self):
    resolver = BotoEndpointResolver(self._endpoint_data())
    services = sorted(resolver.get_available_services())
    expected_services = sorted(['ec2', 's3', 'not-regionalized', 'non-partition', 'merge'])
    self.assertEqual(services, expected_services)