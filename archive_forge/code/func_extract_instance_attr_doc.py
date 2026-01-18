from __future__ import annotations
import inspect
import warnings
from collections import defaultdict
from functools import lru_cache
from typing import Callable
def extract_instance_attr_doc(cls, attr):
    code = inspect.getsource(cls.__init__)
    lines = [line.strip() for line in code.split('\n')]
    i = None
    for i, line in enumerate(lines):
        if line.startswith('self.' + attr + ':') or line.startswith('self.' + attr + ' ='):
            break
    if i is None:
        raise NameError(f'Could not find {attr} in {cls.__name__}')
    start_line = lines.index('"""', i)
    end_line = lines.index('"""', start_line + 1)
    for j in range(i + 1, start_line):
        if lines[j].startswith('self.'):
            raise ValueError(f'Found another attribute before docstring for {attr} in {cls.__name__}: ' + lines[j] + '\n start:' + lines[i])
    doc_string = ' '.join(lines[start_line + 1:end_line])
    return doc_string