import threading
import time
from unittest import mock
from oslo_config import fixture as config
from oslo_serialization import jsonutils
from oslotest import base as test_base
import requests
import webob.dec
import webob.exc
from oslo_middleware import healthcheck
from oslo_middleware.healthcheck import __main__
def test_default_path_not_match(self):
    self._do_test(path='/toto', expected_body=b'Hello, World!!!')