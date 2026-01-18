import copy
import enum
import functools
import logging
import multiprocessing
import shlex
import sys
import threading
from oslo_config import cfg
from oslo_config import types
from oslo_utils import importutils
from oslo_privsep._i18n import _
from oslo_privsep import capabilities
from oslo_privsep import daemon
def set_client_mode(self, enabled):
    if enabled and sys.platform == 'win32':
        raise RuntimeError('Enabling the client_mode is not currently supported on Windows.')
    self.client_mode = enabled