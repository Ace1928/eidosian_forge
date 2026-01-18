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
def fmt_desc(cls, desc):
    desc = textwrap.dedent(desc).strip()
    setattr(cls, 'desc', desc)
    setattr(cls, 'short', desc.splitlines()[0])