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
def get_my_ip():
    """Try to obtain our external ip (from the Pyro4 nameserver's point of view)

    Returns
    -------
    str
        IP address.

    Warnings
    --------
    This tries to sidestep the issue of bogus `/etc/hosts` entries and other local misconfiguration,
    which often mess up hostname resolution.
    If all else fails, fall back to simple `socket.gethostbyname()` lookup.

    """
    import socket
    try:
        from Pyro4.naming import locateNS
        ns = locateNS()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((ns._pyroUri.host, ns._pyroUri.port))
        result, port = s.getsockname()
    except Exception:
        try:
            import commands
            result = commands.getoutput('ifconfig').split('\n')[1].split()[1][5:]
            if len(result.split('.')) != 4:
                raise Exception()
        except Exception:
            result = socket.gethostbyname(socket.gethostname())
    return result