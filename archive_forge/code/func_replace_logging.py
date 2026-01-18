from concurrent import futures
import enum
import errno
import io
import logging as pylogging
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import eventlet
from eventlet import patcher
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_privsep._i18n import _
from oslo_privsep import capabilities
from oslo_privsep import comm
def replace_logging(handler, log_root=None):
    if log_root is None:
        log_root = logging.getLogger(None).logger
    for h in log_root.handlers:
        log_root.removeHandler(h)
    log_root.addHandler(handler)