import functools
import logging
import os
import pkgutil
import re
import traceback
from oslo_utils import strutils
from zunclient import exceptions
from zunclient.i18n import _
def get_available_major_versions():
    matcher = re.compile('v[0-9]*$')
    submodules = pkgutil.iter_modules([os.path.dirname(__file__)])
    available_versions = [name[1:] for loader, name, ispkg in submodules if matcher.search(name)]
    return available_versions