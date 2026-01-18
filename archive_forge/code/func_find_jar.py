import fnmatch
import locale
import os
import re
import stat
import subprocess
import sys
import textwrap
import types
import warnings
from xml.etree import ElementTree
def find_jar(name_pattern, path_to_jar=None, env_vars=(), searchpath=(), url=None, verbose=False, is_regex=False):
    return next(find_jar_iter(name_pattern, path_to_jar, env_vars, searchpath, url, verbose, is_regex))