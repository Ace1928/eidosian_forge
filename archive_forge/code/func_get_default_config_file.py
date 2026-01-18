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
def get_default_config_file():
    config_path = os.path.join(os.path.abspath(os.getcwd()), 'gunicorn.conf.py')
    if os.path.exists(config_path):
        return config_path
    return None