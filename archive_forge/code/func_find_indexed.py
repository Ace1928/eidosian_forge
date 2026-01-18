import ast
import runpy
from inspect import isclass
from pathlib import Path
import pytest
import panel as pn
def find_indexed(index):
    indexed = []
    toctree = False
    for line in index.read_text(encoding='utf-8').split('\n'):
        if line == '```{toctree}':
            toctree = True
        elif not toctree:
            continue
        elif line.startswith('```'):
            toctree = False
        elif line and (not line.startswith(':')):
            if '<' in line:
                line = line[line.index('<') + 1:].rstrip('>')
            indexed.append(line)
    return indexed