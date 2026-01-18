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
@gated_by(CONF.capture_output)
def close_stdio_on_exec():
    fds = [sys.stdin.fileno(), sys.stdout.fileno(), sys.stderr.fileno()]
    for desc in fds:
        fcntl.fcntl(desc, fcntl.F_SETFD, fcntl.FD_CLOEXEC)