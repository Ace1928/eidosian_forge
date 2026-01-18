import ast
import contextlib
import logging
import os
import re
from typing import ClassVar, Sequence
import panel as pn
from .core import OpenFile, get_filesystem_class, split_protocol
from .registry import known_implementations
@property
def storage_options(self):
    """Value of the kwargs box as a dictionary"""
    return ast.literal_eval(self.kwargs.value) or {}