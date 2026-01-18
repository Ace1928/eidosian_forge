from __future__ import annotations
import inspect
import warnings
from collections import defaultdict
from functools import lru_cache
from typing import Callable
def document_cls(cls):
    doc_str = inspect.getdoc(cls)
    if doc_str is None:
        return ('', {}, '')
    tags = {}
    description_lines = []
    mode = 'description'
    for line in doc_str.split('\n'):
        line = line.rstrip()
        if line.endswith(':') and ' ' not in line:
            mode = line[:-1].lower()
            tags[mode] = []
        elif line.split(' ')[0].endswith(':') and (not line.startswith('    ')):
            tag = line[:line.index(':')].lower()
            value = line[line.index(':') + 2:]
            tags[tag] = value
        elif mode == 'description':
            description_lines.append(line if line.strip() else '<br>')
        else:
            if not (line.startswith('    ') or not line.strip()):
                raise SyntaxError(f'Documentation format for {cls.__name__} has format error in line: {line}')
            tags[mode].append(line[4:])
    if 'example' in tags:
        example = '\n'.join(tags['example'])
        del tags['example']
    else:
        example = None
    for key, val in tags.items():
        if isinstance(val, list):
            tags[key] = '<br>'.join(val)
    description = ' '.join(description_lines).replace('\n', '<br>')
    return (description, tags, example)