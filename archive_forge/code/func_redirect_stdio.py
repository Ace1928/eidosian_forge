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
def redirect_stdio(server, capture_output):
    input = [sys.stdin.fileno()]
    output = [sys.stdout.fileno(), sys.stderr.fileno()]
    redirect_to_null(input)
    if capture_output:
        redirect_to_syslog(output, server)
    else:
        redirect_to_null(output)