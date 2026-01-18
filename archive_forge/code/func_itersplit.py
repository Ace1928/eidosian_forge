from __future__ import absolute_import, print_function, division
import re
import operator
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.util.base import Table, asindices
from petl.transform.basics import TransformError
from petl.transform.conversions import convert
def itersplit(source, field, pattern, newfields, include_original, maxsplit, flags):
    it = iter(source)
    prog = re.compile(pattern, flags)
    try:
        hdr = next(it)
    except StopIteration:
        hdr = []
    flds = list(map(text_type, hdr))
    if isinstance(field, int) and field < len(hdr):
        field_index = field
        field = hdr[field_index]
    elif field in flds:
        field_index = flds.index(field)
    else:
        raise ArgumentError('field invalid: must be either field name or index')
    outhdr = list(flds)
    if not include_original:
        outhdr.remove(field)
    if newfields:
        outhdr.extend(newfields)
    yield tuple(outhdr)
    for row in it:
        value = row[field_index]
        if include_original:
            out_row = list(row)
        else:
            out_row = [v for i, v in enumerate(row) if i != field_index]
        out_row.extend(prog.split(value, maxsplit))
        yield tuple(out_row)