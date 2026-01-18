import os
import re
import sys
import typing as t
from pathlib import Path
import zmq
from IPython.core.getipython import get_ipython
from IPython.core.inputtransformer2 import leading_empty_lines
from tornado.locks import Event
from tornado.queues import Queue
from zmq.utils import jsonapi
from .compiler import get_file_name, get_tmp_directory, get_tmp_hash_seed
def disconnect_tcp_socket(self):
    """Disconnect from the tcp socket."""
    self.debugpy_stream.socket.disconnect(self._get_endpoint())
    self.routing_id = None
    self.init_event = Event()
    self.init_event_seq = -1
    self.wait_for_attach = True