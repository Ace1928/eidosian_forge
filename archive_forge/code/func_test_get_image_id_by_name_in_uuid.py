from unittest import mock
import uuid
from saharaclient.api import base as sahara_base
from heat.common import exception
from heat.engine.clients.os import sahara
from heat.tests import common
from heat.tests import utils
def test_get_image_id_by_name_in_uuid(self):
    """Tests the get_image_id function by name in uuid."""
    img_id = str(uuid.uuid4())
    img_name = str(uuid.uuid4())
    self.my_image.id = img_id
    self.my_image.name = img_name
    self.sahara_client.images.get.side_effect = [sahara_base.APIException(error_code=400, error_name='IMAGE_NOT_REGISTERED')]
    self.sahara_client.images.find.return_value = [self.my_image]
    self.assertEqual(img_id, self.sahara_plugin.get_image_id(img_name))
    self.sahara_client.images.get.assert_called_once_with(img_name)
    self.sahara_client.images.find.assert_called_once_with(name=img_name)