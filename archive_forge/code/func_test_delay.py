import os
import socket
import sys
import time
from breezy import config, controldir, errors, tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.osutils import lexists
from breezy.tests import TestCase, TestCaseWithTransport, TestSkipped, features
from breezy.tests.http_server import HttpServer
def test_delay(self):
    sending = FakeSocket()
    receiving = stub_sftp.SocketDelay(sending, 0.1, bandwidth=1000000, really_sleep=False)
    t1 = stub_sftp.SocketDelay.simulated_time
    receiving.send('connect1')
    self.assertEqual(sending.recv(1024), 'connect1')
    t2 = stub_sftp.SocketDelay.simulated_time
    self.assertAlmostEqual(t2 - t1, 0.1)
    receiving.send('connect2')
    self.assertEqual(sending.recv(1024), 'connect2')
    sending.send('hello')
    self.assertEqual(receiving.recv(1024), 'hello')
    t3 = stub_sftp.SocketDelay.simulated_time
    self.assertAlmostEqual(t3 - t2, 0.1)
    sending.send('hello')
    self.assertEqual(receiving.recv(1024), 'hello')
    sending.send('hello')
    self.assertEqual(receiving.recv(1024), 'hello')
    sending.send('hello')
    self.assertEqual(receiving.recv(1024), 'hello')
    t4 = stub_sftp.SocketDelay.simulated_time
    self.assertAlmostEqual(t4, t3)