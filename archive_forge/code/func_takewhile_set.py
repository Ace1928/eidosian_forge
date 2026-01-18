import collections
import inspect
import re
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, Union
from sphinx.application import Sphinx
from sphinx.config import Config as SphinxConfig
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.inspect import stringify_annotation
from sphinx.util.typing import get_type_hints
def takewhile_set(tokens):
    open_braces = 0
    previous_token = None
    while True:
        try:
            token = tokens.popleft()
        except IndexError:
            break
        if token == ', ':
            previous_token = token
            continue
        if not token.strip():
            continue
        if token in keywords:
            tokens.appendleft(token)
            if previous_token is not None:
                tokens.appendleft(previous_token)
            break
        if previous_token is not None:
            yield previous_token
            previous_token = None
        if token == '{':
            open_braces += 1
        elif token == '}':
            open_braces -= 1
        yield token
        if open_braces == 0:
            break