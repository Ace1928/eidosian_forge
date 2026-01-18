from unittest import mock
import uuid
from cinderclient.apiclient import exceptions as apiclient_exception
from cinderclient import exceptions as cinder_exception
from oslo_config import cfg
from oslotest import base
from glance_store.common import cinder_utils
def test_attachment_update(self):
    self.volume_api.attachment_update(self.fake_client, self.fake_attach_id, self.fake_connector)
    self.fake_client.attachments.update.assert_called_once_with(self.fake_attach_id, self.fake_connector)