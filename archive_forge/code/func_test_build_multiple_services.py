import mock
import os
import json
from nose.tools import assert_equal
from tests.unit import unittest
import boto
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
def test_build_multiple_services(self):
    regions = [['mars-west-1', 'moon-darkside-1'], ['mars-west-1']]
    self.resolver.get_all_available_regions.side_effect = regions
    self.resolver.resolve_hostname.side_effect = ['fake-service.mars-west-1.amazonaws.com', 'fake-service.moon-darkside-1.amazonaws.com', 'sample-service.mars-west-1.amazonaws.com']
    services = ['fake-service', 'sample-service']
    endpoints = self.builder.build_static_endpoints(services)
    expected_endpoints = {'fake-service': {'mars-west-1': 'fake-service.mars-west-1.amazonaws.com', 'moon-darkside-1': 'fake-service.moon-darkside-1.amazonaws.com'}, 'sample-service': {'mars-west-1': 'sample-service.mars-west-1.amazonaws.com'}}
    self.assertEqual(endpoints, expected_endpoints)