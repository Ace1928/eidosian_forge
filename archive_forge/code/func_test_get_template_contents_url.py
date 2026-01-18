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
@mock.patch('urllib.request.urlopen')
def test_get_template_contents_url(self, mock_url):
    tmpl = b'{"AWSTemplateFormatVersion" : "2010-09-09", "foo": "bar"}'
    url = 'http://no.where/path/to/a.yaml'
    mock_url.return_value = io.BytesIO(tmpl)
    files, tmpl_parsed = template_utils.get_template_contents(template_url=url)
    self.assertEqual({'AWSTemplateFormatVersion': '2010-09-09', 'foo': 'bar'}, tmpl_parsed)
    self.assertEqual({}, files)
    mock_url.assert_called_with(url)