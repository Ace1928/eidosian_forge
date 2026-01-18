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
@gated_by(CONF.await_child)
@gated_by(CONF.respawn)
def mutually_exclusive():
    sys.stderr.write('--await-child and --respawn are mutually exclusive')
    sys.exit(1)