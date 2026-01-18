import ast
import email.utils
import errno
import fcntl
import html
import importlib
import inspect
import io
import logging
import os
import pwd
import random
import re
import socket
import sys
import textwrap
import time
import traceback
import warnings
from gunicorn.errors import AppImportError
from gunicorn.workers import SUPPORTED_WORKERS
import urllib.parse
def set_owner_process(uid, gid, initgroups=False):
    """ set user and group of workers processes """
    if gid:
        if uid:
            try:
                username = get_username(uid)
            except KeyError:
                initgroups = False
        gid = abs(gid) & 2147483647
        if initgroups:
            os.initgroups(username, gid)
        elif gid != os.getgid():
            os.setgid(gid)
    if uid and uid != os.getuid():
        os.setuid(uid)