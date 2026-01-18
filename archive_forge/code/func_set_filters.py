import ast
import contextlib
import logging
import os
import re
from typing import ClassVar, Sequence
import panel as pn
from .core import OpenFile, get_filesystem_class, split_protocol
from .registry import known_implementations
def set_filters(self, filters=None):
    self.filters = filters
    if filters:
        self.filter_sel.options = filters
        self.filter_sel.value = filters
    else:
        self.filter_sel.options = []
        self.filter_sel.value = []