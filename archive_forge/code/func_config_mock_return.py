import copy
from unittest import mock
import fixtures
from osc_lib.tests import utils as osc_lib_utils
from openstackclient import shell
from openstackclient.tests.unit.integ import base as test_base
from openstackclient.tests.unit import test_shell
def config_mock_return():
    log_file = self.get_temp_file_path('test_log_file')
    cloud2 = test_shell.get_cloud(log_file)
    return ('file.yaml', cloud2)