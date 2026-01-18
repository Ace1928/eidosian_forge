import os
import sys
from datetime import datetime
from functools import partial
import time
from gevent.pool import Pool
from gevent.server import StreamServer
from gevent import hub, monkey, socket, pywsgi
import gunicorn
from gunicorn.http.wsgi import base_environ
from gunicorn.sock import ssl_context
from gunicorn.workers.base_async import AsyncWorker
def handle_usr1(self, sig, frame):
    gevent.spawn(super().handle_usr1, sig, frame)