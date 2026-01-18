import io
import sys
from unittest import mock
from oslo_utils import encodeutils
from requests import Response
import testtools
from glanceclient.common import utils
def create_response_obj_with_req_id(req_id):
    resp = Response()
    resp.headers['x-openstack-request-id'] = req_id
    return resp