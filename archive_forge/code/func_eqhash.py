from __future__ import annotations
def eqhash(o):
    """Call ``obj.__eqhash__``."""
    try:
        return o.__eqhash__()
    except AttributeError:
        return hash(o)