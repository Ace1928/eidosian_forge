import socket
import threading
from io import StringIO
from unittest import skipIf
from dulwich.tests import TestCase
def check_channel_exec_request(self, channel, command):
    self.commands.append(command)
    return True