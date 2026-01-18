import shlex
import sys
import uuid
import hashlib
import collections
import subprocess
import logging
import io
import json
import secrets
import string
import inspect
from html import escape
from functools import wraps
from typing import Union
from dash.types import RendererHooks
def run_command_with_process(cmd):
    is_win = sys.platform == 'win32'
    with subprocess.Popen(shlex.split(cmd, posix=is_win), shell=is_win) as proc:
        proc.wait()
        if proc.poll() is None:
            logger.warning('🚨 trying to terminate subprocess in safe way')
            try:
                proc.communicate()
            except Exception:
                logger.exception('🚨 first try communicate failed')
                proc.kill()
                proc.communicate()