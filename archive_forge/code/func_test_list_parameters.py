import json
import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import configurations
from troveclient.v1 import management
def test_list_parameters(self):

    def side_effect_func(path, config):
        return path
    self.config_params._list = mock.Mock(side_effect=side_effect_func)
    self.assertEqual('/datastores/datastore/versions/version/parameters', self.config_params.parameters('datastore', 'version'))