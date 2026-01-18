import os
import sys
import traceback
from contextlib import contextmanager
from functools import partial
from pprint import pprint
from celery.platforms import signals
from celery.utils.text import WhateverIO
def hfloat(f, p=5):
    """Convert float to value suitable for humans.

    Arguments:
        f (float): The floating point number.
        p (int): Floating point precision (default is 5).
    """
    i = int(f)
    return i if i == f else '{0:.{p}}'.format(f, p=p)