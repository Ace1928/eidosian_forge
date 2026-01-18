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
def start_debugging(argv_0):
    sys.argv[0] = argv_0
    log.debug('sys.argv after patching: {0!r}', sys.argv)
    debugpy.configure(options.config)
    if options.mode == 'listen':
        debugpy.listen(options.address)
    elif options.mode == 'connect':
        debugpy.connect(options.address, access_token=options.adapter_access_token)
    else:
        raise AssertionError(repr(options.mode))
    if options.wait_for_client:
        debugpy.wait_for_client()