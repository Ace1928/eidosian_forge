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
def test_hot_template_same_file(self, mock_url):
    tmpl_file = '/home/my/dir/template.yaml'
    url = 'file://%s' % tmpl_file
    foo_url = 'file:///home/my/dir/foo.yaml'
    contents = b'\nheat_template_version: 2013-05-23\n\noutputs:\n  contents:\n    value:\n      get_file: foo.yaml\n  template:\n    value:\n      get_file: foo.yaml\n'
    mock_url.side_effect = [io.BytesIO(contents), io.BytesIO(b'foo contents')]
    files = template_utils.get_template_contents(template_file=tmpl_file)[0]
    self.assertEqual({foo_url: b'foo contents'}, files)
    mock_url.assert_has_calls([mock.call(url), mock.call(foo_url)])