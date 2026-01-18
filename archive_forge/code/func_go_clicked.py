import ast
import contextlib
import logging
import os
import re
from typing import ClassVar, Sequence
import panel as pn
from .core import OpenFile, get_filesystem_class, split_protocol
from .registry import known_implementations
def go_clicked(self, *_):
    if self.prev_protocol != self.protocol.value or self.prev_kwargs != self.storage_options:
        self._fs = None
        self.prev_protocol = self.protocol.value
        self.prev_kwargs = self.storage_options
    listing = sorted(self.fs.ls(self.url.value, detail=True), key=lambda x: x['name'])
    listing = [l for l in listing if not any((i.match(l['name'].rsplit('/', 1)[-1]) for i in self.ignore))]
    folders = {'📁 ' + o['name'].rsplit('/', 1)[-1]: o['name'] for o in listing if o['type'] == 'directory'}
    files = {'📄 ' + o['name'].rsplit('/', 1)[-1]: o['name'] for o in listing if o['type'] == 'file'}
    if self.filters:
        files = {k: v for k, v in files.items() if any((v.endswith(ext) for ext in self.filters))}
    self.main.set_options(dict(**folders, **files))