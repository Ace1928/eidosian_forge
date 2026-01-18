from unittest import mock
from oslotest import base
from vitrageclient import exceptions as exc
from vitrageclient.tests.utils import get_resources_dir
from vitrageclient.v1.template import Template
def test_add_by_nonexisting_path(self):
    template = Template(mock.Mock())
    self.assertRaises(IOError, template.add, path='non_existing_template_path.yaml')