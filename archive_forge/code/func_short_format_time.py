from __future__ import print_function
import time
import sys
import os
import shutil
import logging
import pprint
from .disk import mkdirp
def short_format_time(t):
    t = _squeeze_time(t)
    if t > 60:
        return '%4.1fmin' % (t / 60.0)
    else:
        return ' %5.1fs' % t