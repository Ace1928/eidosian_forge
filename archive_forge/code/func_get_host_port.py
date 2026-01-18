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
def get_host_port(self):
    """Get the host debugpy port."""
    if self.debugpy_port == -1:
        socket = self.debugpy_stream.socket
        socket.bind_to_random_port('tcp://' + self.debugpy_host)
        self.endpoint = socket.getsockopt(zmq.LAST_ENDPOINT).decode('utf-8')
        socket.unbind(self.endpoint)
        index = self.endpoint.rfind(':')
        self.debugpy_port = self.endpoint[index + 1:]
    return (self.debugpy_host, self.debugpy_port)