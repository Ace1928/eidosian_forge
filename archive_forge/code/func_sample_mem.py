import os
import sys
import traceback
from contextlib import contextmanager
from functools import partial
from pprint import pprint
from celery.platforms import signals
from celery.utils.text import WhateverIO
def sample_mem():
    """Sample RSS memory usage.

    Statistics can then be output by calling :func:`memdump`.
    """
    current_rss = mem_rss()
    _mem_sample.append(current_rss)
    return current_rss