from unittest import mock
from oslo_serialization import jsonutils
import requests
import webob
from keystonemiddleware import ec2_token
from keystonemiddleware.tests.unit import utils
def test_no_key_id(self):
    req = webob.Request.blank('/test')
    req.GET['Signature'] = 'test-signature'
    resp = req.get_response(self.middleware)
    self._validate_ec2_error(resp, 400, 'AuthFailure')