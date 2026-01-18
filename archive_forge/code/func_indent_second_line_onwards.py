import collections
import contextlib
import copy
import inspect
import json
import sys
import textwrap
from typing import (
from itertools import zip_longest
from importlib.metadata import version as importlib_version
from typing import Final
import jsonschema
import jsonschema.exceptions
import jsonschema.validators
import numpy as np
import pandas as pd
from packaging.version import Version
from altair import vegalite
def indent_second_line_onwards(message: str, indent: int=4) -> str:
    modified_lines: List[str] = []
    for idx, line in enumerate(message.split('\n')):
        if idx > 0 and len(line) > 0:
            line = ' ' * indent + line
        modified_lines.append(line)
    return '\n'.join(modified_lines)