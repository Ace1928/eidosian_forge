import importlib.util
import importlib.machinery
import os
import sys
import traceback
from gunicorn import util
from gunicorn.arbiter import Arbiter
from gunicorn.config import Config, get_default_config_file
from gunicorn import debug
def load_config_from_module_name_or_filename(self, location):
    """
        Loads the configuration file: the file is a python file, otherwise raise an RuntimeError
        Exception or stop the process if the configuration file contains a syntax error.
        """
    if location.startswith('python:'):
        module_name = location[len('python:'):]
        cfg = self.get_config_from_module_name(module_name)
    else:
        if location.startswith('file:'):
            filename = location[len('file:'):]
        else:
            filename = location
        cfg = self.get_config_from_filename(filename)
    for k, v in cfg.items():
        if k not in self.cfg.settings:
            continue
        try:
            self.cfg.set(k.lower(), v)
        except Exception:
            print('Invalid value for %s: %s\n' % (k, v), file=sys.stderr)
            sys.stderr.flush()
            raise
    return cfg