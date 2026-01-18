from unittest import mock
from oslotest import base
from vitrageclient import exceptions as exc
from vitrageclient.tests.utils import get_resources_dir
from vitrageclient.v1.template import Template
def test_add_by_path(self):
    template_path = get_resources_dir() + '/template1.yaml'
    template = Template(mock.Mock())
    template.add(path=template_path)