import atexit
import functools
import locale
import logging
import multiprocessing
import os
import traceback
import pathlib
import Pyro4.core
import argparse
from enum import IntEnum
import shutil
import socket
import struct
import collections
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
import uuid
import psutil
import Pyro4
from random import Random
from minerl.env import comms
import minerl.utils.process_watcher
def keep_alive_pyro():

    class KeepAlive(object):

        @Pyro4.expose
        @Pyro4.callback
        def call(self):
            return True
    daemon = Pyro4.core.Daemon()
    callback = KeepAlive()
    daemon.register(callback)
    InstanceManager.add_keep_alive(os.getpid(), callback)
    logger.debug('Client keep-alive server started.')
    daemon.requestLoop()