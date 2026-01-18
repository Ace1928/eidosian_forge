import re
import textwrap
from ast import literal_eval
from inspect import cleandoc
from weakref import WeakKeyDictionary
from parso.python import tree
from parso.cache import parser_cache
from parso import split_lines
def get_sync_comp_fors(comp_for):
    yield comp_for
    last = comp_for.children[-1]
    while True:
        if last.type == 'comp_for':
            yield last.children[1]
        elif last.type == 'sync_comp_for':
            yield last
        elif not last.type == 'comp_if':
            break
        last = last.children[-1]