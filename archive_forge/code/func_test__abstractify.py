import os
import socket
from unittest import mock
from oslotest import base as test_base
from oslo_service import systemd
def test__abstractify(self):
    sock_name = '@fake_socket'
    res = systemd._abstractify(sock_name)
    self.assertEqual('\x00{0}'.format(sock_name[1:]), res)