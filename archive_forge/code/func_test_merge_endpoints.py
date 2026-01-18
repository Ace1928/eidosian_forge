import os
import mock
import boto
from boto.pyami.config import Config
from boto.regioninfo import RegionInfo, load_endpoint_json, merge_endpoints
from boto.regioninfo import load_regions, get_regions, connect
from tests.unit import unittest
def test_merge_endpoints(self):
    defaults = {'ec2': {'us-east-1': 'ec2.us-east-1.amazonaws.com', 'us-west-1': 'ec2.us-west-1.amazonaws.com'}}
    additions = {'s3': {'us-east-1': 's3.amazonaws.com'}, 'ec2': {'us-east-1': 'ec2.auto-resolve.amazonaws.com', 'us-west-2': 'ec2.us-west-2.amazonaws.com'}}
    endpoints = merge_endpoints(defaults, additions)
    self.assertEqual(endpoints, {'ec2': {'us-east-1': 'ec2.auto-resolve.amazonaws.com', 'us-west-1': 'ec2.us-west-1.amazonaws.com', 'us-west-2': 'ec2.us-west-2.amazonaws.com'}, 's3': {'us-east-1': 's3.amazonaws.com'}})