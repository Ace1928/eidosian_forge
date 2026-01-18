import functools
import logging
import multiprocessing
import os
import random
import shlex
import signal
import sys
import time
import warnings
import enum
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_concurrency._i18n import _
def prlimit_args(self):
    """Create a list of arguments for the prlimit command line."""
    args = []
    for limit in self._LIMITS:
        val = getattr(self, limit)
        if val is not None:
            args.append('%s=%s' % (self._LIMITS[limit], val))
    return args