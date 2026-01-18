from unittest import mock
from os_brick.initiator.connectors import remotefs
from os_brick.remotefs import remotefs as remotefs_client
from os_brick.tests.initiator import test_connector
@mock.patch('os_brick.remotefs.remotefs.ScalityRemoteFsClient')
def test_init_with_scality(self, mock_scality_remotefs_client):
    remotefs.RemoteFsConnector('scality', root_helper='sudo')
    self.assertEqual(1, mock_scality_remotefs_client.call_count)