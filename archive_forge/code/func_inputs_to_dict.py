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
def inputs_to_dict(inputs_list):
    inputs = AttributeDict()
    for i in inputs_list:
        inputsi = i if isinstance(i, list) else [i]
        for ii in inputsi:
            id_str = stringify_id(ii['id'])
            inputs[f'{id_str}.{ii['property']}'] = ii.get('value')
    return inputs