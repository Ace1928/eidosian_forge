import io
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from random import randint
from ssl import SSLError
from gunicorn import util
from gunicorn.http.errors import (
from gunicorn.http.wsgi import Response, default_environ
from gunicorn.reloader import reloader_engines
from gunicorn.workers.workertmp import WorkerTmp
def handle_abort(self, sig, frame):
    self.alive = False
    self.cfg.worker_abort(self)
    sys.exit(1)