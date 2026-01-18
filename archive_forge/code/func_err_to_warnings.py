from contextlib import contextmanager
import glob
from importlib import import_module
import io
import itertools
import os.path as osp
import re
import sys
import warnings
import zipfile
import configparser
@staticmethod
@contextmanager
def err_to_warnings():
    try:
        yield
    except BadEntryPoint as e:
        warnings.warn(str(e))