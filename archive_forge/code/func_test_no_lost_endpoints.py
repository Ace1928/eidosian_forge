import mock
import os
import json
from nose.tools import assert_equal
from tests.unit import unittest
import boto
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
def test_no_lost_endpoints():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    old_endpoints_file = os.path.join(data_dir, 'old_endpoints.json')
    with open(old_endpoints_file) as f:
        old_endpoints = json.load(f)
    with open(boto.ENDPOINTS_PATH) as f:
        new_endpoints = json.load(f)
    builder = StaticEndpointBuilder(BotoEndpointResolver(new_endpoints))
    built = builder.build_static_endpoints()
    for service, service_endpoints in old_endpoints.items():
        new_service_endpoints = built.get(service, {})
        for region, regional_endpoint in service_endpoints.items():
            new_regional_endpoint = new_service_endpoints.get(region)
            case = EndpointPreservedTestCase(service, region, regional_endpoint, new_regional_endpoint)
            yield case.run