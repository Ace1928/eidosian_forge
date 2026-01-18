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
def coerce_to_list(obj):
    if not isinstance(obj, (list, tuple)):
        return [obj]
    return obj