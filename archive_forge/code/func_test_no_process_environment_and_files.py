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
def test_no_process_environment_and_files(self):
    files, env = template_utils.process_environment_and_files()
    self.assertEqual({}, env)
    self.assertEqual({}, files)