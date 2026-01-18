import os
import signal
import threading
import time
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import strutils
from os_brick import exception
from os_brick import privileged
@privileged.default.entrypoint
def execute_root(*cmd, **kwargs):
    """NB: Raises processutils.ProcessExecutionError/OSError on failure."""
    return custom_execute(*cmd, shell=False, run_as_root=False, **kwargs)