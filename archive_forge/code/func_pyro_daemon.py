from __future__ import with_statement
from contextlib import contextmanager
import collections.abc
import logging
import warnings
import numbers
from html.entities import name2codepoint as n2cp
import pickle as _pickle
import re
import unicodedata
import os
import random
import itertools
import tempfile
from functools import wraps
import multiprocessing
import shutil
import sys
import subprocess
import inspect
import heapq
from copy import deepcopy
from datetime import datetime
import platform
import types
import numpy as np
import scipy.sparse
from smart_open import open
from gensim import __version__ as gensim_version
def pyro_daemon(name, obj, random_suffix=False, ip=None, port=None, ns_conf=None):
    """Register an object with the Pyro name server.

    Start the name server if not running yet and block until the daemon is terminated.
    The object is registered under `name`, or `name`+ some random suffix if `random_suffix` is set.

    """
    if ns_conf is None:
        ns_conf = {}
    if random_suffix:
        name += '.' + hex(random.randint(0, 16777215))[2:]
    import Pyro4
    with getNS(**ns_conf) as ns:
        with Pyro4.Daemon(ip or get_my_ip(), port or 0) as daemon:
            uri = daemon.register(obj, name)
            ns.remove(name)
            ns.register(name, uri)
            logger.info("%s registered with nameserver (URI '%s')", name, uri)
            daemon.requestLoop()