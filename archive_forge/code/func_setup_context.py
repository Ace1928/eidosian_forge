import os
import sys
import tempfile
import operator
import functools
import itertools
import re
import contextlib
import pickle
import textwrap
import builtins
import pkg_resources
from distutils.errors import DistutilsError
from pkg_resources import working_set
@contextlib.contextmanager
def setup_context(setup_dir):
    temp_dir = os.path.join(setup_dir, 'temp')
    with save_pkg_resources_state():
        with save_modules():
            with save_path():
                hide_setuptools()
                with save_argv():
                    with override_temp(temp_dir):
                        with pushd(setup_dir):
                            __import__('setuptools')
                            yield