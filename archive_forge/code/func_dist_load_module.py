import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
@staticmethod
def dist_load_module(name, path):
    """Load a module from file, required by the abstract class '_Cache'."""
    from .misc_util import exec_mod_from_location
    try:
        return exec_mod_from_location(name, path)
    except Exception as e:
        _Distutils.dist_log(e, stderr=True)
    return None