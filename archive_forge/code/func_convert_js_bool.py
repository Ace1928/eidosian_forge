import html
import os
from os.path import realpath, join, dirname
import sys
import time
import warnings
import webbrowser
import jinja2
from ..frontend_semver import DECKGL_SEMVER
def convert_js_bool(py_bool):
    if type(py_bool) != bool:
        return py_bool
    return 'true' if py_bool else 'false'