import asyncio
import inspect
import os
import signal
import time
from functools import partial
from threading import Thread
import pytest
import zmq
import zmq.asyncio
def new_socket(*args, **kwargs):
    s = context.socket(*args, **kwargs)
    sockets.append(s)
    return s