import argparse
from minerl.env._multiagent import _MultiAgentEnv
from minerl.env.malmo import InstanceManager, MinecraftInstance, malmo_version
from minerl.env import comms
import os
import socket
import struct
import time
import logging
import coloredlogs
def request_interactor(instance, ip):
    sock = get_socket(instance)
    _MultiAgentEnv._TO_MOVE_hello(sock)
    comms.send_message(sock, ('<Interact>' + ip + '</Interact>').encode())
    reply = comms.recv_message(sock)
    ok, = struct.unpack('!I', reply)
    if not ok:
        raise RuntimeError('Failed to start interactor')
    sock.close()