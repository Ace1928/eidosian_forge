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
def test_ex_list(self):
    d = self.driver
    for list_fn in (d.ex_list_addresses, d.ex_list_backendservices, d.ex_list_disktypes, d.ex_list_firewalls, d.ex_list_forwarding_rules, d.ex_list_healthchecks, d.ex_list_networks, d.ex_list_subnetworks, d.ex_list_project_images, d.ex_list_regions, d.ex_list_routes, d.ex_list_snapshots, d.ex_list_targethttpproxies, d.ex_list_targetinstances, d.ex_list_targetpools, d.ex_list_urlmaps, d.ex_list_zones, d.list_images, d.list_locations, d.list_nodes, d.list_sizes, d.list_volumes):
        full_list = [item.name for item in list_fn()]
        li = d.ex_list(list_fn)
        iter_list = [item.name for sublist in li for item in sublist]
        self.assertEqual(full_list, iter_list)
    list_fn = d.ex_list_regions
    for count, sublist in zip((2, 1), d.ex_list(list_fn).page(2)):
        self.assertTrue(len(sublist) == count)
    for sublist in d.ex_list(list_fn).filter('name eq us-central1'):
        self.assertTrue(len(sublist) == 1)
        self.assertEqual(sublist[0].name, 'us-central1')