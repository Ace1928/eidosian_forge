import json
import os
from unittest import mock
import requests
from os_brick import exception
from os_brick.initiator.connectors import scaleio
from os_brick.tests.initiator import test_connector
def test_connect_with_iops_and_bandwidth_limits(self):
    """Successful connect with iops and bandwidth limits"""
    self.fake_connection_properties['bandwidthLimit'] = '500'
    self.fake_connection_properties['iopsLimit'] = '80'
    self.test_connect_volume()