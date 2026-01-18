import socket
import threading
from io import StringIO
from unittest import skipIf
from dulwich.tests import TestCase
def check_channel_request(self, kind, chanid):
    if kind == 'session':
        return paramiko.OPEN_SUCCEEDED
    return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED