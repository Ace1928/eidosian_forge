from __future__ import absolute_import, print_function, division
from petl.compat import next, integer_types, string_types, text_type
import petl.config as config
from petl.errors import ArgumentError, FieldSelectionError
from petl.util.base import Table, expr, fieldnames, Record
from petl.util.parsers import numparser
def iterfieldconvert(source, converters, failonerror, errorvalue, where, pass_row):
    it = iter(source)
    try:
        hdr = next(it)
        flds = list(map(text_type, hdr))
        yield tuple(hdr)
    except StopIteration:
        hdr = flds = []
    converter_functions = dict()
    for k, c in converters.items():
        if not isinstance(k, integer_types):
            try:
                k = flds.index(k)
            except ValueError:
                raise FieldSelectionError(k)
        assert isinstance(k, int), 'expected integer, found %r' % k
        if callable(c):
            converter_functions[k] = c
        elif isinstance(c, string_types):
            converter_functions[k] = methodcaller(c)
        elif isinstance(c, (tuple, list)) and isinstance(c[0], string_types):
            methnm = c[0]
            methargs = c[1:]
            converter_functions[k] = methodcaller(methnm, *methargs)
        elif isinstance(c, dict):
            converter_functions[k] = dictconverter(c)
        elif c is None:
            pass
        else:
            raise ArgumentError('unexpected converter specification on field %r: %r' % (k, c))

    def transform_value(i, v, *args):
        if i not in converter_functions:
            return v
        else:
            try:
                return converter_functions[i](v, *args)
            except Exception as e:
                if failonerror == 'inline':
                    return e
                elif failonerror:
                    raise e
                else:
                    return errorvalue
    if pass_row:

        def transform_row(_row):
            return tuple((transform_value(i, v, _row) for i, v in enumerate(_row)))
    else:

        def transform_row(_row):
            return tuple((transform_value(i, v) for i, v in enumerate(_row)))
    if isinstance(where, string_types):
        where = expr(where)
    elif where is not None:
        assert callable(where), 'expected callable for "where" argument, found %r' % where
    if pass_row or where:
        it = (Record(row, flds) for row in it)
    if where is None:
        for row in it:
            yield transform_row(row)
    else:
        for row in it:
            if where(row):
                yield transform_row(row)
            else:
                yield row