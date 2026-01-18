import json
import os
import re
import sys
from importlib.util import find_spec
import pydevd
from _pydevd_bundle import pydevd_runpy as runpy
import debugpy
from debugpy.common import log
from debugpy.server import api
import codecs;
import json;
import sys;
import attach_pid_injected;
def run_file():
    target = options.target
    start_debugging(target)
    if os.path.isfile(target):
        dir = os.path.dirname(target)
        sys.path.insert(0, dir)
    else:
        log.debug('Not a file: {0!r}', target)
    log.describe_environment('Pre-launch environment:')
    log.info('Running file {0!r}', target)
    runpy.run_path(target, run_name='__main__')