import inspect
import os.path
import sys
from _pydev_bundle._pydev_tipper_common import do_find
from _pydevd_bundle.pydevd_utils import hasattr_checked, dir_checked
from inspect import getfullargspec
def generate_tip(data, log=None):
    data = data.replace('\n', '')
    if data.endswith('.'):
        data = data.rstrip('.')
    f, mod, parent, foundAs = Find(data, log)
    tips = generate_imports_tip_for_module(mod)
    return (f, tips)