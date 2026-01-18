from unittest import mock
from heat.common import exception as heat_exception
from heat.engine.clients.os import monasca as client_plugin
from heat.tests import common
from heat.tests import utils
@mock.patch.object(client_plugin.MonascaClientPlugin, 'client')
def test_get_notification_not_found(self, client_monasca):
    self._client.notifications.get.side_effect = client_plugin.monasca_exc.NotFound
    client_monasca.return_value = self._client
    ex = self.assertRaises(heat_exception.EntityNotFound, self.client_plugin.get_notification, self.sample_uuid)
    msg = 'The Monasca Notification (%(name)s) could not be found.' % {'name': self.sample_uuid}
    self.assertEqual(msg, str(ex))
    self._client.notifications.get.assert_called_once_with(notification_id=self.sample_uuid)