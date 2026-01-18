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
def noqa_line_for(self, line_number: int) -> str | None:
    """Retrieve the line which will be used to determine noqa."""
    return self._noqa_line_mapping.get(line_number)