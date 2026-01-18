from __future__ import print_function
import os
import io
import time
import functools
import collections
import collections.abc
import numpy as np
import requests
import IPython
def get_ioloop():
    from tornado.ioloop import IOLoop
    ipython = IPython.get_ipython()
    if ipython and hasattr(ipython, 'kernel'):
        return IOLoop.instance()