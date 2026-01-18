from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.util.base import Table
def iterunpack(source, field, newfields, include_original, missing):
    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        hdr = []
    flds = list(map(text_type, hdr))
    if field in flds:
        field_index = flds.index(field)
    elif isinstance(field, int) and field < len(flds):
        field_index = field
        field = flds[field_index]
    else:
        raise ArgumentError('field invalid: must be either field name or index')
    outhdr = list(flds)
    if not include_original:
        outhdr.remove(field)
    if isinstance(newfields, (list, tuple)):
        outhdr.extend(newfields)
        nunpack = len(newfields)
    elif isinstance(newfields, int):
        nunpack = newfields
        newfields = [text_type(field) + text_type(i + 1) for i in range(newfields)]
        outhdr.extend(newfields)
    elif newfields is None:
        nunpack = 0
    else:
        raise ArgumentError('newfields argument must be list or tuple of field names, or int (number of values to unpack)')
    yield tuple(outhdr)
    for row in it:
        value = row[field_index]
        if include_original:
            out_row = list(row)
        else:
            out_row = [v for i, v in enumerate(row) if i != field_index]
        nvals = len(value)
        if nunpack > 0:
            if nvals >= nunpack:
                newvals = value[:nunpack]
            else:
                newvals = list(value) + [missing] * (nunpack - nvals)
            out_row.extend(newvals)
        yield tuple(out_row)