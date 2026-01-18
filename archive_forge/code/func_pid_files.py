import argparse
import fcntl
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from oslo_config import cfg
from oslo_utils import units
from glance.common import config
from glance.i18n import _
def pid_files(server, pid_file):
    pid_files = []
    if pid_file:
        if os.path.exists(os.path.abspath(pid_file)):
            pid_files = [os.path.abspath(pid_file)]
    elif os.path.exists('/var/run/glance/%s.pid' % server):
        pid_files = ['/var/run/glance/%s.pid' % server]
    for pid_file in pid_files:
        pid = int(open(pid_file).read().strip())
        yield (pid_file, pid)