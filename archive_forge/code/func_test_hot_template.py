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
def test_hot_template(self, mock_url):
    tmpl_file = '/home/my/dir/template.yaml'
    url = 'file:///home/my/dir/template.yaml'
    foo_url = 'file:///home/my/dir/foo.yaml'
    bar_url = 'file:///home/my/dir/bar.yaml'

    def side_effect(args):
        if url == args:
            return io.BytesIO(self.hot_template)
        if foo_url == args:
            return io.BytesIO(self.foo_template)
        if bar_url == args:
            return io.BytesIO(self.bar_template)
    mock_url.side_effect = side_effect
    files, tmpl_parsed = template_utils.get_template_contents(template_file=tmpl_file)
    self.assertEqual(yaml.safe_load(self.bar_template.decode('utf-8')), json.loads(files.get('file:///home/my/dir/bar.yaml')))
    self.assertEqual({'heat_template_version': '2013-05-23', 'resources': {'foo': {'type': 'OS::Type1', 'properties': {'config': {'get_file': 'file:///home/my/dir/bar.yaml'}}}}}, json.loads(files.get('file:///home/my/dir/foo.yaml')))
    self.assertEqual({'heat_template_version': '2013-05-23', 'resources': {'resource1': {'type': 'OS::Heat::Stack', 'properties': {'template': {'get_file': 'file:///home/my/dir/foo.yaml'}}}}}, tmpl_parsed)
    mock_url.assert_has_calls([mock.call(foo_url), mock.call(url), mock.call(bar_url)], any_order=True)