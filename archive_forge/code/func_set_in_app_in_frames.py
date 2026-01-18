import base64
import json
import linecache
import logging
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import namedtuple
from copy import copy
from decimal import Decimal
from numbers import Real
from datetime import datetime
from functools import partial
import sentry_sdk
from sentry_sdk._compat import PY2, PY33, PY37, implements_str, text_type, urlparse
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import DEFAULT_MAX_VALUE_LENGTH
def set_in_app_in_frames(frames, in_app_exclude, in_app_include, project_root=None):
    if not frames:
        return None
    for frame in frames:
        current_in_app = frame.get('in_app')
        if current_in_app is not None:
            continue
        module = frame.get('module')
        if _module_in_list(module, in_app_include):
            frame['in_app'] = True
            continue
        if _module_in_list(module, in_app_exclude):
            frame['in_app'] = False
            continue
        abs_path = frame.get('abs_path')
        if abs_path is None:
            continue
        if _is_external_source(abs_path):
            frame['in_app'] = False
            continue
        if _is_in_project_root(abs_path, project_root):
            frame['in_app'] = True
            continue
    return frames