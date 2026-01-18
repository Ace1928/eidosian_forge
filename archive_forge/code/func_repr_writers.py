from __future__ import annotations
from kombu.utils.eventio import ERR, READ, WRITE
from kombu.utils.functional import reprcall
def repr_writers(h):
    """Return description of pending writers."""
    return [f'({fd}){_rcb(cb)}->{repr_flag(WRITE)}' for fd, cb in h.writers.items()]