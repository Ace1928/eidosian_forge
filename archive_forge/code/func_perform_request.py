from unittest import mock
import uuid
from oslotest import base as test_base
import statsd
import webob.dec
import webob.exc
from oslo_middleware import stats
def perform_request(self, app, path, method):
    req = webob.Request.blank(path, method=method)
    return req.get_response(app)