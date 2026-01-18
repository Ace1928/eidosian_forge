import _imp
import os
import re
import sys
import warnings
from functools import partial
from .errors import DistutilsPlatformError
from sysconfig import (
def parse_config_h(fp, g=None):
    return sysconfig_parse_config_h(fp, vars=g)