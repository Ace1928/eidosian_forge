from functools import partial
import sys
from eventlet import hubs, greenthread
from eventlet.greenio import GreenSocket
import eventlet.wsgi
import greenlet
from gunicorn.workers.base_async import AsyncWorker
from gunicorn.sock import ssl_wrap_socket
def patch_sendfile():
    if not hasattr(GreenSocket, 'sendfile'):
        GreenSocket.sendfile = _eventlet_socket_sendfile