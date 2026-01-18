from unittest import mock
import uuid
from saharaclient.api import base as sahara_base
from heat.common import exception
from heat.engine.clients.os import sahara
from heat.tests import common
from heat.tests import utils
def test_get_image_id_not_found(self):
    """Tests the get_image_id function while image is not found."""
    img_name = str(uuid.uuid4())
    self.my_image.name = img_name
    self.sahara_client.images.get.side_effect = [sahara_base.APIException(error_code=400, error_name='IMAGE_NOT_REGISTERED')]
    self.sahara_client.images.find.return_value = []
    self.assertRaises(exception.EntityNotFound, self.sahara_plugin.get_image_id, img_name)
    self.sahara_client.images.get.assert_called_once_with(img_name)
    self.sahara_client.images.find.assert_called_once_with(name=img_name)