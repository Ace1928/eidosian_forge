import glob
import http.client
import queue
from unittest import mock
from unittest.mock import mock_open
from os_brick import exception
from os_brick.initiator.connectors import lightos
from os_brick.initiator import linuxscsi
from os_brick.privileged import lightos as priv_lightos
from os_brick.tests.initiator import test_connector
from os_brick import utils
def test_monitor_message_queue_delete(self):
    message_queue = queue.Queue()
    connection = {'uuid': '123'}
    message_queue.put(('delete', connection))
    lightos_db = {'123': 'fake_connection'}
    self.connector.monitor_message_queue(message_queue, lightos_db)
    self.assertEqual(len(lightos_db), 0)