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
def test_get_template_file_nonextant(self):
    nonextant_file = '/template/dummy/file/path/and/name.yaml'
    ex = self.assertRaises(error.URLError, template_utils.get_template_contents, nonextant_file)
    self.assertEqual("<urlopen error [Errno 2] No such file or directory: '%s'>" % nonextant_file, str(ex))