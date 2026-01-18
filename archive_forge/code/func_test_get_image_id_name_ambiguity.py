from unittest import mock
import uuid
from saharaclient.api import base as sahara_base
from heat.common import exception
from heat.engine.clients.os import sahara
from heat.tests import common
from heat.tests import utils
def test_get_image_id_name_ambiguity(self):
    """Tests the get_image_id function while name ambiguity ."""
    img_name = 'ambiguity_name'
    self.my_image.name = img_name
    self.sahara_client.images.get.side_effect = sahara_base.APIException()
    self.sahara_client.images.find.return_value = [self.my_image, self.my_image]
    self.assertRaises(exception.PhysicalResourceNameAmbiguity, self.sahara_plugin.get_image_id, img_name)
    self.sahara_client.images.find.assert_called_once_with(name=img_name)