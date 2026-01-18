from unittest import mock
import oslo_messaging as messaging
from heat.rpc import api as rpc_api
from heat.rpc import listener_client as rpc_client
from heat.tests import common
@mock.patch('heat.common.messaging.get_rpc_client', return_value=mock.Mock())
def test_engine_alive_timeout(self, rpc_client_method):
    mock_rpc_client = rpc_client_method.return_value
    mock_prepare_method = mock_rpc_client.prepare
    mock_prepare_client = mock_prepare_method.return_value
    mock_cnxt = mock.Mock()
    listener_client = rpc_client.EngineListenerClient('engine-007')
    rpc_client_method.assert_called_once_with(version=rpc_client.EngineListenerClient.BASE_RPC_API_VERSION, topic=rpc_api.LISTENER_TOPIC, server='engine-007')
    mock_prepare_method.assert_called_once_with(timeout=2)
    self.assertEqual(mock_prepare_client, listener_client._client, 'Failed to create RPC client')
    mock_prepare_client.call.side_effect = messaging.MessagingTimeout('too slow')
    ret = listener_client.is_alive(mock_cnxt)
    self.assertFalse(ret)
    mock_prepare_client.call.assert_called_once_with(mock_cnxt, 'listening')