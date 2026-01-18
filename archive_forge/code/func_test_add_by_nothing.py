from unittest import mock
from oslotest import base
from vitrageclient import exceptions as exc
from vitrageclient.tests.utils import get_resources_dir
from vitrageclient.v1.template import Template
def test_add_by_nothing(self):
    template = Template(mock.Mock())
    self.assertRaises(exc.CommandError, template.add)