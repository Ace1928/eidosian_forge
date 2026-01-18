from __future__ import absolute_import, print_function, division
import operator
from petl.compat import text_type
from petl.util.base import Table, asindices, Record
def iterproblems(table, constraints, expected_header):
    outhdr = ('name', 'row', 'field', 'value', 'error')
    yield outhdr
    it = iter(table)
    try:
        actual_header = next(it)
    except StopIteration:
        actual_header = []
    if expected_header is None:
        flds = list(map(text_type, actual_header))
    else:
        expected_flds = list(map(text_type, expected_header))
        actual_flds = list(map(text_type, actual_header))
        try:
            assert expected_flds == actual_flds
        except Exception as e:
            yield ('__header__', 0, None, None, type(e).__name__)
        flds = expected_flds
    local_constraints = normalize_constraints(constraints, flds)
    for constraint in local_constraints:
        if 'getter' not in constraint:
            if 'field' in constraint:
                indices = asindices(flds, constraint['field'])
                getter = operator.itemgetter(*indices)
                constraint['getter'] = getter
    expected_len = len(flds)
    for i, row in enumerate(it):
        row = tuple(row)
        l = None
        try:
            l = len(row)
            assert l == expected_len
        except Exception as e:
            yield ('__len__', i + 1, None, l, type(e).__name__)
        row = Record(row, flds)
        for constraint in local_constraints:
            name = constraint.get('name', None)
            field = constraint.get('field', None)
            assertion = constraint.get('assertion', None)
            test = constraint.get('test', None)
            getter = constraint.get('getter', lambda x: x)
            try:
                target = getter(row)
            except Exception as e:
                yield (name, i + 1, field, None, type(e).__name__)
            else:
                value = target if field else None
                if test is not None:
                    try:
                        test(target)
                    except Exception as e:
                        yield (name, i + 1, field, value, type(e).__name__)
                if assertion is not None:
                    try:
                        assert assertion(target)
                    except Exception as e:
                        yield (name, i + 1, field, value, type(e).__name__)