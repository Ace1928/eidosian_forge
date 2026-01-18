from __future__ import annotations
from kombu.utils.eventio import ERR, READ, WRITE
from kombu.utils.functional import reprcall
def repr_flag(flag):
    """Return description of event loop flag."""
    return '{}{}{}'.format('R' if flag & READ else '', 'W' if flag & WRITE else '', '!' if flag & ERR else '')