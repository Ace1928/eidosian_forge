import contextlib
import os
import sys
from typing import List, Optional, Type, Union
from . import i18n, option, osutils, trace
from .lazy_import import lazy_import
import breezy
from breezy import (
from . import errors, registry
from .hooks import Hooks
from .i18n import gettext
from .plugin import disable_plugins, load_plugins, plugin_name
def apply_coveraged(the_callable, *args, **kwargs):
    import coverage
    cov = coverage.Coverage()
    config_file = cov.config.config_file
    os.environ['COVERAGE_PROCESS_START'] = config_file
    cov.start()
    try:
        return exception_to_return_code(the_callable, *args, **kwargs)
    finally:
        cov.stop()
        cov.save()