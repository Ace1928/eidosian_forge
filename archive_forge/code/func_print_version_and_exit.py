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
def print_version_and_exit(switch, it):
    print(debugpy.__version__)
    sys.exit(0)