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
def split_callback_id(callback_id):
    if callback_id.startswith('..'):
        return [split_callback_id(oi) for oi in callback_id[2:-2].split('...')]
    id_, prop = callback_id.rsplit('.', 1)
    return {'id': id_, 'property': prop}