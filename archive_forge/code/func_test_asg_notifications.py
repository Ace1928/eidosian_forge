import kombu
from oslo_config import cfg
from oslo_messaging._drivers import common
from oslo_messaging import transport
import requests
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_asg_notifications(self):
    stack_identifier = self.stack_create(template=self.asg_template)
    for output in self.client.stacks.get(stack_identifier).outputs:
        if output['output_key'] == 'scale_dn_url':
            scale_down_url = output['output_value']
        else:
            scale_up_url = output['output_value']
    notifications = []
    handler = NotificationHandler(stack_identifier.split('/')[0], ASG_NOTIFICATIONS)
    with self.conn.Consumer(self.queue, callbacks=[handler.process_message], auto_declare=False):
        requests.post(scale_up_url, verify=self.verify_cert)
        self.assertTrue(test.call_until_true(20, 0, self.consume_events, handler, 2))
        notifications += handler.notifications
        handler.clear()
        requests.post(scale_down_url, verify=self.verify_cert)
        self.assertTrue(test.call_until_true(20, 0, self.consume_events, handler, 2))
        notifications += handler.notifications
    self.assertEqual(2, notifications.count(ASG_NOTIFICATIONS[0]))
    self.assertEqual(2, notifications.count(ASG_NOTIFICATIONS[1]))