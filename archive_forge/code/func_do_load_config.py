import importlib.util
import importlib.machinery
import os
import sys
import traceback
from gunicorn import util
from gunicorn.arbiter import Arbiter
from gunicorn.config import Config, get_default_config_file
from gunicorn import debug
def do_load_config(self):
    """
        Loads the configuration
        """
    try:
        self.load_default_config()
        self.load_config()
    except Exception as e:
        print('\nError: %s' % str(e), file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)