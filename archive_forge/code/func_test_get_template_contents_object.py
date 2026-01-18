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
def test_get_template_contents_object(self):
    tmpl = '{"AWSTemplateFormatVersion" : "2010-09-09", "foo": "bar"}'
    url = 'http://no.where/path/to/a.yaml'
    self.object_requested = False

    def object_request(method, object_url):
        self.object_requested = True
        self.assertEqual('GET', method)
        self.assertEqual('http://no.where/path/to/a.yaml', object_url)
        return tmpl
    files, tmpl_parsed = template_utils.get_template_contents(template_object=url, object_request=object_request)
    self.assertEqual({'AWSTemplateFormatVersion': '2010-09-09', 'foo': 'bar'}, tmpl_parsed)
    self.assertEqual({}, files)
    self.assertTrue(self.object_requested)