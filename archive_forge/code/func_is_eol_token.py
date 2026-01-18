from __future__ import annotations
import argparse
import ast
import functools
import logging
import tokenize
from typing import Any
from typing import Generator
from typing import List
from typing import Tuple
from flake8 import defaults
from flake8 import utils
from flake8._compat import FSTRING_END
from flake8._compat import FSTRING_MIDDLE
from flake8.plugins.finder import LoadedPlugin
def is_eol_token(token: tokenize.TokenInfo) -> bool:
    """Check if the token is an end-of-line token."""
    return token[0] in NEWLINE or token[4][token[3][1]:].lstrip() == '\\\n'