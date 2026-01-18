import importlib.util
import importlib.machinery
import os
import sys
import traceback
from gunicorn import util
from gunicorn.arbiter import Arbiter
from gunicorn.config import Config, get_default_config_file
from gunicorn import debug
def get_config_from_filename(self, filename):
    if not os.path.exists(filename):
        raise RuntimeError("%r doesn't exist" % filename)
    ext = os.path.splitext(filename)[1]
    try:
        module_name = '__config__'
        if ext in ['.py', '.pyc']:
            spec = importlib.util.spec_from_file_location(module_name, filename)
        else:
            msg = 'configuration file should have a valid Python extension.\n'
            util.warn(msg)
            loader_ = importlib.machinery.SourceFileLoader(module_name, filename)
            spec = importlib.util.spec_from_file_location(module_name, filename, loader=loader_)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
    except Exception:
        print('Failed to read config file: %s' % filename, file=sys.stderr)
        traceback.print_exc()
        sys.stderr.flush()
        sys.exit(1)
    return vars(mod)