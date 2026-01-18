import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_callback_without_progress(self):
    """Callback is called with 0 if 'progress' attribute is missing."""
    statuses = ['active', 'deleting', 'deleting', 'deleting', 'deleted']
    res = self._fake_resource(statuses=statuses)
    callback = mock.Mock()
    result = resource.wait_for_delete(mock.Mock(), res, interval=1, wait=5, callback=callback)
    self.assertEqual(result, res)
    callback.assert_has_calls([mock.call(0)] * 3)