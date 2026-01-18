import json
import os
from unittest import mock
import requests
from os_brick import exception
from os_brick.initiator.connectors import scaleio
from os_brick.tests.initiator import test_connector
def handle_scaleio_request(self, url, *args, **kwargs):
    """Fake REST server"""
    api_call = url.split(':', 2)[2].split('/', 1)[1].replace('api/', '')
    if 'setMappedSdcLimits' in api_call:
        self.assertNotIn('iops_limit', kwargs['data'])
        if 'iopsLimit' not in kwargs['data']:
            self.assertIn('bandwidthLimitInKbps', kwargs['data'])
        elif 'bandwidthLimitInKbps' not in kwargs['data']:
            self.assertIn('iopsLimit', kwargs['data'])
        else:
            self.assertIn('bandwidthLimitInKbps', kwargs['data'])
            self.assertIn('iopsLimit', kwargs['data'])
    try:
        return self.mock_calls[api_call]
    except KeyError:
        return self.error_404