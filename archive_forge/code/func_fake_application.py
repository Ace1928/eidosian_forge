from unittest import mock
import uuid
from oslotest import base as test_base
import statsd
import webob.dec
import webob.exc
from oslo_middleware import stats
@webob.dec.wsgify
def fake_application(req):
    return 'Hello, World'