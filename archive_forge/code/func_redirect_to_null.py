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
def redirect_to_null(fds):
    with open(os.devnull, 'r+b') as nullfile:
        for desc in fds:
            try:
                os.dup2(nullfile.fileno(), desc)
            except OSError:
                pass