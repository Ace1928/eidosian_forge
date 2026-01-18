import os
import sys
import re
from collections.abc import Iterator
from warnings import warn
from looseversion import LooseVersion
import numpy as np
import textwrap
def dict_diff(dold, dnew, indent=0):
    """Helper to log what actually changed from old to new values of
    dictionaries.

    typical use -- log difference for hashed_inputs
    """
    try:
        dnew, dold = (dict(dnew), dict(dold))
    except Exception:
        return textwrap.indent(f'Diff between nipype inputs failed:\n* Cached inputs: {dold}\n* New inputs: {dnew}', ' ' * indent)
    new_keys = set(dnew.keys())
    old_keys = set(dold.keys())
    diff = []
    if new_keys - old_keys:
        diff += ['  * keys not previously seen: %s' % (new_keys - old_keys)]
    if old_keys - new_keys:
        diff += ['  * keys not presently seen: %s' % (old_keys - new_keys)]
    if diff:
        diff.insert(0, 'Dictionaries had differing keys:')
    diffkeys = len(diff)

    def _shorten(value):
        if isinstance(value, str) and len(value) > 50:
            return f'{value[:10]}...{value[-10:]}'
        if isinstance(value, (tuple, list)) and len(value) > 10:
            return tuple(list(value[:2]) + ['...'] + list(value[-2:]))
        return value

    def _uniformize(val):
        if isinstance(val, dict):
            return {k: _uniformize(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return tuple((_uniformize(el) for el in val))
        return val
    for k in new_keys.intersection(old_keys):
        new = _uniformize(dnew[k])
        old = _uniformize(dold[k])
        if new != old:
            diff += ['  * %s: %r != %r' % (k, _shorten(new), _shorten(old))]
    if len(diff) > diffkeys:
        diff.insert(diffkeys, 'Some dictionary entries had differing values:')
    return textwrap.indent('\n'.join(diff), ' ' * indent)