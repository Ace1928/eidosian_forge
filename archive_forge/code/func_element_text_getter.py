from __future__ import absolute_import, print_function, division
from operator import attrgetter
import itertools
from petl.compat import string_types, text_type
from petl.util.base import Table, fieldnames, iterpeek
from petl.io.sources import read_source_from_arg
from petl.io.text import totext
def element_text_getter(missing):

    def _get(v):
        if len(v) > 1:
            return tuple((e.text for e in v))
        elif len(v) == 1:
            return v[0].text
        else:
            return missing
    return _get