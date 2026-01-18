import errno
import fcntl
import os
from oslo_log import log as logging
import select
import signal
import socket
import ssl
import struct
import sys
import termios
import time
import tty
from urllib import parse as urlparse
import websocket
from zunclient.common.apiclient import exceptions as acexceptions
from zunclient.common.websocketclient import exceptions
def do_exec(zunclient, url, container_id, exec_id, escape, close_wait):
    if url.startswith('ws://') or url.startswith('wss://'):
        try:
            wscls = ExecClient(zunclient=zunclient, url=url, exec_id=exec_id, id=container_id, escape=escape, close_wait=close_wait)
            wscls.connect()
            wscls.handle_resize()
            wscls.start_loop()
        except exceptions.ContainerWebSocketException as e:
            print('%(e)s:%(container)s' % {'e': e, 'container': container_id})
    else:
        raise exceptions.InvalidWebSocketLink(container_id)