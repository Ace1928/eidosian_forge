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
def get_caller_name():
    stack = inspect.stack()
    for s in stack:
        if s.function == '<module>':
            return s.frame.f_locals.get('__name__', '__main__')
    return '__main__'