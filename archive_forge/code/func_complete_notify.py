import ast
import cmd
import signal
import socket
import sys
import termios
from os_ken import cfg
from os_ken.lib import rpc
def complete_notify(self, text, line, begidx, endidx):
    return self._complete_peer(text, line, begidx, endidx)