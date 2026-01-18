from __future__ import annotations
import ast
import collections
import os
import re
import sys
import token
import tokenize
from dataclasses import dataclass
from types import CodeType
from typing import (
from coverage import env
from coverage.bytecode import code_objects
from coverage.debug import short_stack
from coverage.exceptions import NoSource, NotPython
from coverage.misc import join_regex, nice_pair
from coverage.phystokens import generate_tokens
from coverage.types import TArc, TLineNo
def first_lines(self, linenos: Iterable[TLineNo]) -> set[TLineNo]:
    """Map the line numbers in `linenos` to the correct first line of the
        statement.

        Returns a set of the first lines.

        """
    return {self.first_line(l) for l in linenos}