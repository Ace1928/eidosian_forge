import itertools
import os
import pprint
import select
import socket
import threading
import time
import fixtures
from keystoneauth1 import exceptions
import prometheus_client
from requests import exceptions as rexceptions
import testtools.content
from openstack.tests.unit import base
def assert_prometheus_stat(self, name, value, labels=None):
    sample_value = self._registry.get_sample_value(name, labels)
    self.assertEqual(sample_value, value)