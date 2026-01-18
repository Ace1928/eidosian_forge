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
@property
def proc_name(self):
    pn = self.settings['proc_name'].get()
    if pn is not None:
        return pn
    else:
        return self.settings['default_proc_name'].get()