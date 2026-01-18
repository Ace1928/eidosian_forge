import sys
import datetime
import unittest
from unittest import mock
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gce import (
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
def test_format_metadata(self):
    in_md = [{'key': 'k0', 'value': 'v0'}, {'key': 'k1', 'value': 'v1'}]
    out_md = self.driver._format_metadata('fp', in_md)
    self.assertTrue('fingerprint' in out_md)
    self.assertEqual(out_md['fingerprint'], 'fp')
    self.assertTrue('items' in out_md)
    self.assertEqual(len(out_md['items']), 2)
    self.assertTrue(out_md['items'][0]['key'] in ['k0', 'k1'])
    self.assertTrue(out_md['items'][0]['value'] in ['v0', 'v1'])
    in_md = [{'k0': 'v0'}, {'k1': 'v1'}]
    out_md = self.driver._format_metadata('fp', in_md)
    self.assertTrue('fingerprint' in out_md)
    self.assertEqual(out_md['fingerprint'], 'fp')
    self.assertTrue('items' in out_md)
    self.assertEqual(len(out_md['items']), 2)
    self.assertTrue(out_md['items'][0]['key'] in ['k0', 'k1'])
    self.assertTrue(out_md['items'][0]['value'] in ['v0', 'v1'])
    in_md = {'key': 'k0', 'value': 'v0'}
    out_md = self.driver._format_metadata('fp', in_md)
    self.assertTrue('fingerprint' in out_md)
    self.assertEqual(out_md['fingerprint'], 'fp')
    self.assertTrue('items' in out_md)
    self.assertEqual(len(out_md['items']), 1, out_md)
    self.assertEqual(out_md['items'][0]['key'], 'k0')
    self.assertEqual(out_md['items'][0]['value'], 'v0')
    in_md = {'k0': 'v0'}
    out_md = self.driver._format_metadata('fp', in_md)
    self.assertTrue('fingerprint' in out_md)
    self.assertEqual(out_md['fingerprint'], 'fp')
    self.assertTrue('items' in out_md)
    self.assertEqual(len(out_md['items']), 1)
    self.assertEqual(out_md['items'][0]['key'], 'k0')
    self.assertEqual(out_md['items'][0]['value'], 'v0')
    in_md = {'k0': 'v0', 'k1': 'v1', 'k2': 'v2'}
    out_md = self.driver._format_metadata('fp', in_md)
    self.assertTrue('fingerprint' in out_md)
    self.assertEqual(out_md['fingerprint'], 'fp')
    self.assertTrue('items' in out_md)
    self.assertEqual(len(out_md['items']), 3)
    keys = [x['key'] for x in out_md['items']]
    vals = [x['value'] for x in out_md['items']]
    keys.sort()
    vals.sort()
    self.assertEqual(keys, ['k0', 'k1', 'k2'])
    self.assertEqual(vals, ['v0', 'v1', 'v2'])
    in_md = {'items': [{'key': 'k0', 'value': 'v0'}, {'key': 'k1', 'value': 'v1'}]}
    out_md = self.driver._format_metadata('fp', in_md)
    self.assertTrue('fingerprint' in out_md)
    self.assertEqual(out_md['fingerprint'], 'fp')
    self.assertTrue('items' in out_md)
    self.assertEqual(len(out_md['items']), 2)
    self.assertTrue(out_md['items'][0]['key'] in ['k0', 'k1'])
    self.assertTrue(out_md['items'][0]['value'] in ['v0', 'v1'])
    in_md = {'items': 'foo'}
    self.assertRaises(ValueError, self.driver._format_metadata, 'fp', in_md)
    in_md = {'items': {'key': 'k1', 'value': 'v0'}}
    self.assertRaises(ValueError, self.driver._format_metadata, 'fp', in_md)
    in_md = ['k0', 'v1']
    self.assertRaises(ValueError, self.driver._format_metadata, 'fp', in_md)