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
def test_get_template_contents_file_none_existing(self):
    files, tmpl_parsed = template_utils.get_template_contents(existing=True)
    self.assertIsNone(tmpl_parsed)
    self.assertEqual({}, files)