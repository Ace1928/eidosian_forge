import json
import os
import re
import sys
from importlib.util import find_spec
import pydevd
from _pydevd_bundle import pydevd_runpy as runpy
import debugpy
from debugpy.common import log
from debugpy.server import api
import codecs;
import json;
import sys;
import attach_pid_injected;
def set_target(kind, parser=lambda x: x, positional=False):

    def do(arg, it):
        options.target_kind = kind
        target = parser(arg if positional else next(it))
        if isinstance(target, bytes):
            try:
                target = target.decode(sys.getfilesystemencoding())
            except UnicodeDecodeError:
                try:
                    target = target.decode('utf-8')
                except UnicodeDecodeError:
                    import locale
                    target = target.decode(locale.getpreferredencoding(False))
        options.target = target
    return do