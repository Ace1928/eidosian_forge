import contextlib
import itertools
import logging
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from oslo_utils import timeutils
from oslo_utils import uuidutils
from zake import fake_client
from taskflow.conductors import backends as conductors
from taskflow import engines
from taskflow.jobs import backends as boards
from taskflow.patterns import linear_flow
from taskflow.persistence import backends as persistence
from taskflow.persistence import models
from taskflow import task
from taskflow.utils import threading_utils
def generate_conductor(client, saver, name=NAME):
    """Creates a conductor thread with the given name prefix."""
    real_name = '%s_conductor' % name
    jb = boards.fetch(name, JOBBOARD_CONF, client=client, persistence=saver)
    conductor = conductors.fetch('blocking', real_name, jb, engine='parallel', wait_timeout=SCAN_DELAY)

    def run():
        jb.connect()
        with contextlib.closing(jb):
            conductor.run()
    return (threading_utils.daemon_thread(target=run), conductor.stop)