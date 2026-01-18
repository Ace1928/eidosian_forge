import ast
import contextlib
import logging
import os
import re
from typing import ClassVar, Sequence
import panel as pn
from .core import OpenFile, get_filesystem_class, split_protocol
from .registry import known_implementations
def selection_changed(self, *_):
    if self.urlpath is None:
        return
    if self.fs.isdir(self.urlpath):
        self.url.value = self.fs._strip_protocol(self.urlpath)
    self.go_clicked()