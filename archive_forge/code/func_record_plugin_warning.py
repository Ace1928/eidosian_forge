import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def record_plugin_warning(warning_message):
    trace.mutter(warning_message)
    return warning_message