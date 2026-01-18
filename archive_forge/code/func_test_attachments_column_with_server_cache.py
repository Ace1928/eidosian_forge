from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume
def test_attachments_column_with_server_cache(self):
    _volume = volume_fakes.create_one_volume()
    server_id = _volume.attachments[0]['server_id']
    device = _volume.attachments[0]['device']
    fake_server = mock.Mock()
    fake_server.name = 'fake-server-name'
    server_cache = {server_id: fake_server}
    col = volume.AttachmentsColumn(_volume.attachments, server_cache)
    self.assertEqual('Attached to %s on %s ' % ('fake-server-name', device), col.human_readable())
    self.assertEqual(_volume.attachments, col.machine_readable())