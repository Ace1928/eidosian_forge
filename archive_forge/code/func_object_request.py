import json
import tempfile
from unittest import mock
import io
from oslo_serialization import base64
import testtools
from testtools import matchers
from urllib import error
import yaml
from heatclient.common import template_utils
from heatclient.common import utils
from heatclient import exc
def object_request(method, object_url):
    self.object_requested = True
    self.assertEqual('GET', method)
    self.assertTrue(object_url.startswith('http://no.where/path/to/'))
    if object_url == url:
        return tmpl
    else:
        return '{"heat_template_version": "2016-04-08"}'