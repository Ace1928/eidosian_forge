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
def inputs_to_vals(inputs):
    return [[ii.get('value') for ii in i] if isinstance(i, list) else i.get('value') for i in inputs]