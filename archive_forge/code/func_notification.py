import ast
import cmd
import signal
import socket
import sys
import termios
from os_ken import cfg
from os_ken.lib import rpc
def notification(self, n):
    print('NOTIFICATION from %s %s' % (self._name, n))