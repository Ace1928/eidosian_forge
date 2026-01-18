import mock
import os
import json
from nose.tools import assert_equal
from tests.unit import unittest
import boto
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
def test_get_all_regions_on_non_regional_service(self):
    resolver = BotoEndpointResolver(self._endpoint_data())
    regions = sorted(resolver.get_all_available_regions('not-regionalized'))
    expected_regions = sorted(['us-foo', 'us-bar', 'eu-baz'])
    self.assertEqual(regions, expected_regions)