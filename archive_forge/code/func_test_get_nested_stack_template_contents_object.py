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
def test_get_nested_stack_template_contents_object(self):
    tmpl = '{"heat_template_version": "2016-04-08","resources": {"FooBar": {"type": "foo/bar.yaml"}}}'
    url = 'http://no.where/path/to/a.yaml'
    self.object_requested = False

    def object_request(method, object_url):
        self.object_requested = True
        self.assertEqual('GET', method)
        self.assertTrue(object_url.startswith('http://no.where/path/to/'))
        if object_url == url:
            return tmpl
        else:
            return '{"heat_template_version": "2016-04-08"}'
    files, tmpl_parsed = template_utils.get_template_contents(template_object=url, object_request=object_request)
    self.assertEqual(files['http://no.where/path/to/foo/bar.yaml'], '{"heat_template_version": "2016-04-08"}')
    self.assertTrue(self.object_requested)