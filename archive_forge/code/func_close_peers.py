import ast
import cmd
import signal
import socket
import sys
import termios
from os_ken import cfg
from os_ken.lib import rpc
def close_peers():
    for peer in peers.values():
        peer.socket.close()