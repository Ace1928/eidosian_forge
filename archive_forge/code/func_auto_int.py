import argparse
import copy
import grp
import inspect
import os
import pwd
import re
import shlex
import ssl
import sys
import textwrap
from gunicorn import __version__, util
from gunicorn.errors import ConfigError
from gunicorn.reloader import reloader_engines
def auto_int(_, x):
    if re.match('0(\\d)', x, re.IGNORECASE):
        x = x.replace('0', '0o', 1)
    return int(x, 0)