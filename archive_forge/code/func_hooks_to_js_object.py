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
def hooks_to_js_object(hooks: Union[RendererHooks, None]) -> str:
    if hooks is None:
        return ''
    hook_str = ','.join((f'{key}: {val}' for key, val in hooks.items()))
    return f'{{{hook_str}}}'