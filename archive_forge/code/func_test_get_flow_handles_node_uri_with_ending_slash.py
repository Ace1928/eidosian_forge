import sys
from unittest import mock
import urllib.error
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_utils import units
import taskflow
import glance.async_.flows.api_image_import as import_flow
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance import context
from glance.domain import ExtraProperties
from glance import gateway
import glance.tests.utils as test_utils
from cursive import exception as cursive_exception
def test_get_flow_handles_node_uri_with_ending_slash(self):
    test_uri = 'file:///some/where/'
    expected_uri = '{0}{1}'.format(test_uri, IMAGE_ID1)
    self._pass_uri(uri=test_uri, file_uri=expected_uri, import_req=self.gd_task_input['import_req'])
    self._pass_uri(uri=test_uri, file_uri=expected_uri, import_req=self.wd_task_input['import_req'])