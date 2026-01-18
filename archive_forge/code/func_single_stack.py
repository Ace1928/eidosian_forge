from __future__ import annotations
import logging # isort:skip
from ..transform import stack
def single_stack(stackers, spec, **kw):
    if spec in kw:
        raise ValueError("Stack property '%s' cannot appear in keyword args" % spec)
    lengths = {len(x) for x in kw.values() if isinstance(x, (list, tuple))}
    if len(lengths) > 0:
        if len(lengths) != 1:
            raise ValueError('Keyword argument sequences for broadcasting must all be the same lengths. Got lengths: %r' % sorted(list(lengths)))
        if lengths.pop() != len(stackers):
            raise ValueError('Keyword argument sequences for broadcasting must be the same length as stackers')
    s = []
    _kw = []
    for i, val in enumerate(stackers):
        d = {'name': val}
        s.append(val)
        d[spec] = stack(*s)
        for k, v in kw.items():
            if isinstance(v, (list, tuple)):
                d[k] = v[i]
            else:
                d[k] = v
        _kw.append(d)
    return _kw