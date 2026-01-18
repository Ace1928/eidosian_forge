import json
import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import configurations
from troveclient.v1 import management
def test_get_parameter(self):

    def side_effect_func(path):
        return path
    self.config_params._get = mock.Mock(side_effect=side_effect_func)
    self.assertEqual('/datastores/datastore/versions/version/parameters/key', self.config_params.get_parameter('datastore', 'version', 'key'))